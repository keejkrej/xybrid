package ai.xybrid.reactnative

import ai.xybrid.XybridModel
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

internal class XybridHandleTable {
  private val nextHandle = AtomicLong(1)
  private val models = ConcurrentHashMap<Long, XybridModel>()

  fun insert(model: XybridModel): Long {
    val handle = nextHandle.getAndIncrement()
    models[handle] = model
    return handle
  }

  fun get(handle: Long): XybridModel =
    models[handle] ?: throw IllegalArgumentException("Unknown or disposed model handle: $handle")

  fun remove(handle: Long) {
    models.remove(handle)?.destroy()
  }

  fun clear() {
    val handles = models.keys.toList()
    for (handle in handles) {
      remove(handle)
    }
  }
}
