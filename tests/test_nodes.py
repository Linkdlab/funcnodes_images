import unittest
import funcnodes_images as fnimg
import numpy as np
from PIL import Image
import tempfile


class TestNodes(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.img_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.img = Image.fromarray(self.img_arr)

    async def test_from_bytes(self):
        temp_jpg = tempfile.NamedTemporaryFile(suffix=".jpg")
        temp_jpg.close()
        self.img.save(temp_jpg.name, quality=95)
        with open(temp_jpg.name, "rb") as f:
            jpeg_bytes = f.read()

        refimg = Image.open(temp_jpg.name)

        frombytes = fnimg.nodes.FromBytes()
        frombytes.get_input("data").value = jpeg_bytes
        await frombytes
        img: fnimg.PillowImageFormat = frombytes.get_output("img").value
        self.assertIsInstance(img, fnimg.PillowImageFormat)

        np.testing.assert_equal(img.to_array(), np.array(refimg))

    async def test_resize(self):
        resize = fnimg.nodes.ResizeImage()
        resize.get_input("img").value = fnimg.PillowImageFormat(self.img)
        resize.get_input("width").value = 50
        resize.get_input("height").value = 50

        import asyncio

        await asyncio.sleep(1)
        img: fnimg.PillowImageFormat = resize.get_output("resized_img").value
        self.assertEqual(img.to_array().shape, (50, 50, 3))

    async def test_crop(self):
        crop = fnimg.nodes.CropImage()
        crop.get_input("img").value = fnimg.PillowImageFormat(self.img)
        crop.get_input("x1").value = 10
        crop.get_input("y1").value = 10
        crop.get_input("x2").value = 90
        crop.get_input("y2").value = 90

        await crop
        img: fnimg.PillowImageFormat = crop.get_output("cropped_img").value
        self.assertEqual(img.to_array().shape, (80, 80, 3))

    async def test_scale(self):
        scale = fnimg.nodes.ScaleImage()
        scale.get_input("img").value = fnimg.PillowImageFormat(self.img)
        scale.get_input("scale").value = 0.5

        await scale
        img: fnimg.PillowImageFormat = scale.get_output("scaled_img").value
        self.assertEqual(img.to_array().shape, (50, 50, 3))
