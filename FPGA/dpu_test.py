from ctypes import *
from typing import List
import cv2
import vart
import xir
import threading
import sys
from skimage.transform import resize
from skimage.io import imread

from imutils.video import FPS

from ICAI_Pose_inference import ICAIPose
from overwrite_queue import OverwriteQueue

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):
    # video_capture_str = 'mediasrcbin media-device=/dev/media0 v4l2src0::io-mode=dmabuf v4l2src0::stride-align=256  ! video/x-raw, width=256, height=256, format=NV12, framerate=30/1 ! videoconvert! appsink'

    ICAIPose_xmodel = 'ICAIPose.xmodel'

    ICAIPose_graph = xir.Graph.deserialize(ICAIPose_xmodel)
    ICAIPose_subgraphs = get_child_subgraph_dpu(ICAIPose_graph)
    runner = vart.Runner.create_runner(ICAIPose_subgraphs[0], "run")
    dpu_ICAIPose = ICAIPose(runner)
    dpu_ICAIPose.start()

    # frame = cv2.imread('data/multi.jpg')
    # frame = cv2.resize(frame, (256,256))

    frames = [cv2.imread('data/im1.jpg'), cv2.imread('data/im2.jpg'), cv2.imread('data/im3.jpg')]



    for im in frames:
        res = dpu_ICAIPose.process(im)
    # res = resize(res, (1024,1024))

        # cv2.imshow('frame', res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
