require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "xybrid-react-native"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.license      = { :type => "Apache-2.0", :file => "../../LICENSE" }
  s.homepage     = "https://github.com/xybrid-ai/xybrid"
  s.authors      = { "Xybrid AI" => "support@xybrid.ai" }
  s.platforms    = { :ios => "13.0" }
  s.source       = { :git => "https://github.com/xybrid-ai/xybrid.git", :tag => "v#{s.version}" }
  s.source_files = "ios/**/*.{h,m,mm,swift}"
  s.preserve_paths = "ios/Frameworks/XybridFFI.xcframework"
  s.vendored_frameworks = "ios/Frameworks/XybridFFI.xcframework"
  s.dependency "React-Core"
  s.dependency "React-Codegen"
  s.pod_target_xcconfig = {
    "DEFINES_MODULE" => "YES",
    "OTHER_LDFLAGS" => "$(inherited) -lc++",
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20"
  }
  s.frameworks = [
    "Accelerate",
    "CoreML",
    "Metal",
    "MetalPerformanceShaders",
    "MetalPerformanceShadersGraph",
    "Security"
  ]
end
