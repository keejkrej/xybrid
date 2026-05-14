package ai.xybrid.reactnative

import ai.xybrid.XybridEnvelope
import ai.xybrid.XybridGenerationConfig
import ai.xybrid.XybridModelLoader
import ai.xybrid.XybridThermalState
import ai.xybrid.clearBatteryLevel
import ai.xybrid.initSdkCacheDir
import ai.xybrid.setBatteryLevel
import ai.xybrid.setBinding
import ai.xybrid.setThermalState
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.sun.jna.Library
import com.sun.jna.Native
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File

class XybridReactNativeModule(
  private val reactContext: ReactApplicationContext,
) : NativeXybridSpec(reactContext) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
  private val handles = XybridHandleTable()
  @Volatile private var initialized = false

  override fun getName(): String = NAME

  override fun initialize(options: ReadableMap?, promise: Promise) {
    try {
      if (!initialized) {
        synchronized(this) {
          if (!initialized) {
            setBinding("react-native")
            val cacheDir = options?.getString("cacheDir")
              ?: File(reactContext.filesDir, "xybrid/models").absolutePath
            initSdkCacheDir(cacheDir)
            registerPlatformObservers(reactContext.applicationContext)
            initialized = true
          }
        }
      }
      promise.resolve(null)
    } catch (error: Throwable) {
      promise.reject("xybrid_initialize_error", error.message, error)
    }
  }

  override fun setApiKey(apiKey: String) {
    LibC.INSTANCE.setenv("XYBRID_API_KEY", apiKey, 1)
  }

  override fun loadModel(source: ReadableMap, requestId: String, promise: Promise) {
    scope.launch {
      val modelId = source.getString("value")
      try {
        emitProgress(requestId, modelId, 0.0, "starting")
        val loader = loaderFromSource(source)
        emitProgress(requestId, modelId, 0.1, "loading")
        val model = loader.load()
        val handle = handles.insert(model)
        emitProgress(requestId, modelId, 1.0, "ready")
        promise.resolve(Arguments.createMap().apply {
          putDouble("handle", handle.toDouble())
        })
      } catch (error: Throwable) {
        emitLoadError(requestId, error.message ?: "Failed to load model")
        promise.reject("xybrid_load_error", error.message, error)
      }
    }
  }

  override fun runModel(handle: Double, envelope: ReadableMap, config: ReadableMap?, promise: Promise) {
    scope.launch {
      try {
        val model = handles.get(handle.toLong())
        val result = model.run(envelopeFromReadableMap(envelope), generationConfigFromReadableMap(config))
        promise.resolve(Arguments.createMap().apply {
          putBoolean("success", result.success)
          result.text?.let { putString("text", it) }
          result.audioBytes?.let {
            putString("audioBase64", Base64.encodeToString(it, Base64.NO_WRAP))
          }
          result.embedding?.let { putArray("embedding", floatListToArray(it)) }
          putDouble("latencyMs", result.latencyMs.toDouble())
        })
      } catch (error: Throwable) {
        promise.reject("xybrid_run_error", error.message, error)
      }
    }
  }

  override fun voices(handle: Double, promise: Promise) {
    try {
      val array = Arguments.createArray()
      handles.get(handle.toLong()).voices()?.forEach { voice ->
        array.pushMap(Arguments.createMap().apply {
          putString("id", voice.id)
          putString("name", voice.name)
          voice.gender?.let { putString("gender", it) }
          voice.language?.let { putString("language", it) }
          voice.style?.let { putString("style", it) }
        })
      }
      promise.resolve(array)
    } catch (error: Throwable) {
      promise.reject("xybrid_voices_error", error.message, error)
    }
  }

  override fun defaultVoiceId(handle: Double, promise: Promise) {
    try {
      promise.resolve(handles.get(handle.toLong()).defaultVoiceId())
    } catch (error: Throwable) {
      promise.reject("xybrid_voice_error", error.message, error)
    }
  }

  override fun disposeModel(handle: Double) {
    handles.remove(handle.toLong())
  }

  override fun isModelCached(modelId: String, promise: Promise) {
    val cacheDir = File(reactContext.filesDir, "xybrid/models")
    promise.resolve(File(cacheDir, modelId).exists())
  }

  override fun invalidate() {
    handles.clear()
    super.invalidate()
  }

  private fun loaderFromSource(source: ReadableMap): XybridModelLoader {
    val kind = source.getString("kind") ?: "registry"
    val value = source.getString("value") ?: throw IllegalArgumentException("Model source value is required")
    return when (kind) {
      "registry" -> XybridModelLoader.fromRegistry(value)
      "bundle" -> XybridModelLoader.fromBundle(value)
      "directory" -> XybridModelLoader.fromDirectory(value)
      "huggingface" -> XybridModelLoader.fromHuggingface(value)
      else -> throw IllegalArgumentException("Unsupported model source kind: $kind")
    }
  }

  private fun envelopeFromReadableMap(map: ReadableMap): XybridEnvelope {
    return when (val kind = map.getString("kind")) {
      "text" -> XybridEnvelope.Text(
        map.getString("text") ?: "",
        map.getString("voiceId"),
        if (map.hasKey("speed") && !map.isNull("speed")) map.getDouble("speed") else null,
      )
      "audio" -> XybridEnvelope.Audio(
        Base64.decode(map.getString("audioBase64") ?: "", Base64.DEFAULT),
        map.optionalUInt("sampleRate", 16000u),
        map.optionalUInt("channels", 1u),
      )
      "embedding" -> XybridEnvelope.Embedding(floatListFromArray(map.getArray("embedding")))
      else -> throw IllegalArgumentException("Unsupported envelope kind: $kind")
    }
  }

  private fun generationConfigFromReadableMap(map: ReadableMap?): XybridGenerationConfig? {
    if (map == null) return null
    return XybridGenerationConfig(
      maxTokens = map.optionalUInt("maxTokens"),
      temperature = map.optionalFloat("temperature"),
      topP = map.optionalFloat("topP"),
      minP = map.optionalFloat("minP"),
      topK = map.optionalUInt("topK"),
      repetitionPenalty = map.optionalFloat("repetitionPenalty"),
      stopSequences = if (map.hasKey("stopSequences") && !map.isNull("stopSequences")) map.getArray("stopSequences")?.let { array ->
        List(array.size()) { i -> array.getString(i) ?: "" }
      } else null,
    )
  }

  private fun emitProgress(requestId: String, modelId: String?, progress: Double, phase: String) {
    emitXybridLoadProgress(Arguments.createMap().apply {
      putString("requestId", requestId)
      modelId?.let { putString("modelId", it) }
      putDouble("progress", progress)
      putString("phase", phase)
    })
  }

  private fun emitLoadError(requestId: String, message: String) {
    emitXybridLoadError(Arguments.createMap().apply {
      putString("requestId", requestId)
      putString("message", message)
    })
  }

  private fun registerPlatformObservers(appContext: Context) {
    val batteryReceiver = object : BroadcastReceiver() {
      override fun onReceive(received: Context, intent: Intent) {
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        if (level < 0 || scale <= 0) {
          clearBatteryLevel()
          return
        }
        val pct = ((level * 100) / scale).coerceIn(0, 100)
        setBatteryLevel(pct.toUByte())
      }
    }
    appContext.registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      val pm = appContext.getSystemService(Context.POWER_SERVICE) as PowerManager
      setThermalState(thermalStatusToXybrid(pm.currentThermalStatus))
      pm.addThermalStatusListener { status ->
        setThermalState(thermalStatusToXybrid(status))
      }
    }
  }

  private fun thermalStatusToXybrid(status: Int): XybridThermalState = when (status) {
    PowerManager.THERMAL_STATUS_NONE,
    PowerManager.THERMAL_STATUS_LIGHT -> XybridThermalState.NORMAL
    PowerManager.THERMAL_STATUS_MODERATE -> XybridThermalState.WARM
    PowerManager.THERMAL_STATUS_SEVERE -> XybridThermalState.HOT
    else -> XybridThermalState.CRITICAL
  }

  companion object {
    const val NAME = "XybridReactNative"
  }
}

private fun ReadableMap.optionalUInt(key: String, defaultValue: UInt? = null): UInt? =
  if (hasKey(key) && !isNull(key)) getDouble(key).toUInt() else defaultValue

private fun ReadableMap.optionalFloat(key: String): Float? =
  if (hasKey(key) && !isNull(key)) getDouble(key).toFloat() else null

private fun floatListFromArray(array: ReadableArray?): List<Float> {
  if (array == null) return emptyList()
  return List(array.size()) { i -> array.getDouble(i).toFloat() }
}

private fun floatListToArray(values: List<Float>): WritableArray =
  Arguments.createArray().apply {
    values.forEach { pushDouble(it.toDouble()) }
  }

private interface LibC : Library {
  fun setenv(name: String, value: String, overwrite: Int): Int

  companion object {
    val INSTANCE: LibC = Native.load("c", LibC::class.java)
  }
}
