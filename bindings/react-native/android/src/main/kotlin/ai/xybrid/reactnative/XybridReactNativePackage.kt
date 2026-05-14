package ai.xybrid.reactnative

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider

class XybridReactNativePackage : BaseReactPackage() {
  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? =
    if (name == XybridReactNativeModule.NAME) {
      XybridReactNativeModule(reactContext)
    } else {
      null
    }

  override fun getReactModuleInfoProvider(): ReactModuleInfoProvider = ReactModuleInfoProvider {
    mapOf(
      XybridReactNativeModule.NAME to ReactModuleInfo(
        XybridReactNativeModule.NAME,
        XybridReactNativeModule.NAME,
        false,
        false,
        false,
        true,
      ),
    )
  }
}
