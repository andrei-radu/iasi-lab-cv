import cv2


def create_tracker(tracker_type="kcf"):
    """Create a CSRT or KCF tracker, supporting different OpenCV builds."""
    if tracker_type == "csrt":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()

    elif tracker_type == "kcf":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()

    raise RuntimeError(
        f"{tracker_type} tracker is unavailable. "
        "Try an OpenCV build with contrib tracking support."
    )
