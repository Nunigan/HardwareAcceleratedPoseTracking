/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>


#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


const int TNUM = 1;

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
array<bool, TNUM> is_running;
bool is_displaying = true;

// comparison algorithm for priority_queue
class Compare {
 public:
  bool operator()(const pair<int, Mat>& n1, const pair<int, Mat>& n2) const {
    return n1.first > n2.first;
  }
};

queue<pair<int, Mat>> read_queue;  // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare>
    display_queue;        // display queue
mutex mtx_read_queue;     // mutex of read queue
mutex mtx_display_queue;  // mutex of display queue
int read_index = 0;       // frame index of input video
int display_index = 0;    // frame index to display

GraphInfo shapes;


/**
 * @brief Run DPU and ARM Tasks for SSD, and put image into display queue
 *
 * @param task_conv - pointer to SSD CONV Task
 * @param is_running - status flag of RunSSD thread
 *
 * @return none
 */
void RunICAIPose(vart::Runner* runner, bool& is_running) {
  // get out tensors and shapes
  auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());
  auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
  auto input_scale = 0.5;
  auto output_scale = get_output_scale(runner->get_output_tensors()[0]);
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int inSize = shapes.inTensorList[0].size;
  int size = shapes.outTensorList[0].size;
  int batchSize = in_dims[0];
  int batch = inputTensors[0]->get_shape().at(0);

  printf("size = %d", size);
  printf("batch = %d", batch);

  int8_t* out = new int8_t[size * batch];
  int8_t* imageInputs = new int8_t[inSize * batch];

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

  // Run detection for images in read queue
  while (is_running) {
    // Get an image from read queue
    int index;
    Mat img;
    mtx_read_queue.lock();
    if (read_queue.empty()) {
      mtx_read_queue.unlock();
      if (is_reading) {
        continue;
      } else {
        is_running = false;
        break;
      }
    } else {
      index = read_queue.front().first;
      img = read_queue.front().second;
      read_queue.pop();
      mtx_read_queue.unlock();
    }

    float mean[3] = {103.939, 116.779, 123.6};
    Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
    resize(img, image2, Size(inWidth, inHeight), 0, 0, INTER_LINEAR);

    for (int h = 0; h < inHeight; h++) {
      for (int w = 0; w < inWidth; w++) {
        for (int c = 0; c < 3; c++) {
          imageInputs[h * inWidth * 3 + w * 3 + c] =
              (int8_t)((image2.at<Vec3b>(h, w)[c] - mean[c]) * input_scale);
        }
      }
    }

    // input/output prepare
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, inputTensors[0].get()));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        out, outputTensors[0].get()));

    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());
    // execute
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    inputsPtr.clear();
    outputsPtr.clear();
    inputs.clear();
    outputs.clear();

    // uint8_t img_out[256*256];


    // for(int i =  0; i<1; i++)
    // {
    //   for(int j = 0; j<256; ++j)
    //   {
    //       for(int k = 0; j<256; ++j)
    //       {
    //         img_out[j*256+k] = out[j+256*k];
    //       }
    //   }
    // }

    // img = Mat(512, 512, CV_8UC1, out);


    // Put image into display queue
    mtx_display_queue.lock();
    display_queue.push(make_pair(index, img));
    mtx_display_queue.unlock();
  }

  // delete[] imageInputs;
}


/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool& is_reading) {
  while (is_reading) {
    Mat img;
    if (read_queue.size() < 1) {
      if (!video.read(img)) {
        cout << "Video end." << endl;
        is_reading = false;
        break;
      }
      mtx_read_queue.lock();
      read_queue.push(make_pair(read_index++, img));
      mtx_read_queue.unlock();
    } else {
      usleep(5000);
    }
  }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool& is_displaying) {
  Mat image(512, 512, CV_8UC1);
  imshow("Video Analysis @Xilinx DPU", image);
  while (is_displaying) {
    mtx_display_queue.lock();
    if (display_queue.empty()) {
      if (any_of(is_running.begin(), is_running.end(),
                 [](bool cond) { return cond; })) {
        mtx_display_queue.unlock();
        usleep(20);
      } else {
        is_displaying = false;
        break;
      }
    } else if (display_index == display_queue.top().first) {
      // Display image
      imshow("Video Analysis @Xilinx DPU", display_queue.top().second);
      display_index++;
      display_queue.pop();
      mtx_display_queue.unlock();
      if (waitKey(1) == 'q') {
        is_reading = false;
        for (int i = 0; i < TNUM; ++i) {
          is_running[i] = false;
        }

        is_displaying = false;
        break;
      }
    } else {
      mtx_display_queue.unlock();
    }
  }
}

int main(int argc, char* argv[]) {
  // Check args
  if (argc != 3) {
    cout << "Usage of ICAIPose demo: ./ICAIPose [video_file] "
            "[model_file]"
         << endl;
    return -1;
  }

  // Initializations
  string file_name = argv[1];
  cout << "Detect video: " << file_name << endl;
  video.open(file_name);
  if (!video.isOpened()) {
    cout << "Failed to open video: " << file_name;
    return -1;
  }

  auto graph = xir::Graph::deserialize(argv[2]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "ICAIPose should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  // create runner
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  // auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
  // auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  // auto runner3 = vart::Runner::create_runner(subgraph[0], "run");
  // auto runner4 = vart::Runner::create_runner(subgraph[0], "run");
  // auto runner5 = vart::Runner::create_runner(subgraph[0], "run");

  // get input/output tensor shapes
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  // // Run tasks for ICAIpose
  vector<thread> threads(TNUM);
  is_running.fill(true);
  threads[0] = thread(RunICAIPose, runner.get(), ref(is_running[0]));
  // threads[1] = thread(RunICAIPose, runner1.get(), ref(is_running[1]));
  // threads[2] = thread(RunICAIPose, runner2.get(), ref(is_running[2]));
  // threads[3] = thread(RunICAIPose, runner3.get(), ref(is_running[3]));
  // threads[4] = thread(RunICAIPose, runner4.get(), ref(is_running[4]));
  // threads[5] = thread(RunICAIPose, runner5.get(), ref(is_running[5]));
  threads.push_back(thread(Read, ref(is_reading)));
  threads.push_back(thread(Display, ref(is_displaying)));

  for (int i = 0; i < 2 + TNUM; ++i) {
    threads[i].join();
  }

  video.release();
  return 0;
}
