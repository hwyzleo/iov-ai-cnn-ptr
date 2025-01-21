import ctypes
import cv2
import numpy as np
import os

# 加载SIPL库
sipl_lib = ctypes.CDLL('libNvSIPLCamera.so')


# 定义摄像头配置结构体
class NvSIPLCameraConfig(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),        # 图像宽度
        ("height", ctypes.c_uint32),       # 图像高度
        ("format", ctypes.c_uint32),       # 图像格式
        ("frameRate", ctypes.c_float),     # 帧率
        ("exposureTime", ctypes.c_float),  # 曝光时间
        ("gain", ctypes.c_float),          # 增益
        ("triggerMode", ctypes.c_uint32),  # 触发模式
        ("bufferCount", ctypes.c_uint32)   # 缓冲区数量
    ]

    def __init__(self):
        super().__init__()
        self.width = 1920
        self.height = 1080
        self.format = 0  # 根据实际支持的格式定义
        self.frameRate = 30.0
        self.exposureTime = 33.33  # 毫秒
        self.gain = 1.0
        self.triggerMode = 0  # 连续模式
        self.bufferCount = 4


# 摄像头管理类
class CameraManager:
    def __init__(self):
        self.camera_handle = self.init_camera()
        self.camera_config = NvSIPLCameraConfig()
        self.configure_camera()
        self.start_capture()

    @staticmethod
    def init_camera():
        camera_handle = ctypes.c_void_p()
        status = sipl_lib.NvSIPLCameraCreate(ctypes.byref(camera_handle))
        if status != 0:
            raise Exception("Failed to create camera handle")
        return camera_handle

    def configure_camera(self):
        config = NvSIPLCameraConfig()
        status = sipl_lib.NvSIPLCameraSetConfig(self.camera_handle, ctypes.byref(config))
        if status != 0:
            raise Exception("Failed to configure camera")

    def start_capture(self):
        status = sipl_lib.NvSIPLCameraStartCapture(self.camera_handle)
        if status != 0:
            raise Exception("Failed to start capture")

    def get_frame(self):
        try:
            frame_data = ctypes.c_void_p()
            status = sipl_lib.NvSIPLCameraGetFrame(self.camera_handle, ctypes.byref(frame_data))
            if status != 0:
                return False, None
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.camera_config.height, self.camera_config.width, 3))
            return True, frame
        except Exception:
            return False, None

    @staticmethod
    def get(prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0  # 返回默认帧率
        return 0

    def release(self):
        sipl_lib.NvSIPLCameraStopCapture(self.camera_handle)
        sipl_lib.NvSIPLCameraDestroy(self.camera_handle)