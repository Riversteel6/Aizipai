package com.example.ai

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class LosslessFrame(
    val pixels: ByteArray,
    val width: Int,
    val height: Int,
    val rowBytes: Int,
)

object LosslessFrameFactory {
    fun fromBitmap(bitmap: Bitmap): LosslessFrame {
        require(bitmap.config == Bitmap.Config.ARGB_8888) { "frame_must_be_argb_8888" }
        require(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN) {
            "unsupported_android_byte_order"
        }
        val size = Math.multiplyExact(bitmap.rowBytes, bitmap.height)
        val buffer = ByteBuffer.allocate(size).order(ByteOrder.nativeOrder())
        bitmap.copyPixelsToBuffer(buffer)
        return LosslessFrame(
            pixels = buffer.array(),
            width = bitmap.width,
            height = bitmap.height,
            rowBytes = bitmap.rowBytes,
        )
    }
}
