#import "XybridReactNative.h"
#import "xybrid_react_native-Swift.h"

@implementation XybridReactNative {
  XybridReactNativeAdapter *_adapter;
}

RCT_EXPORT_MODULE(XybridReactNative)

- (instancetype)init
{
  if (self = [super init]) {
    _adapter = [XybridReactNativeAdapter new];
    __weak typeof(self) weakSelf = self;
    [_adapter setProgressEmitter:^(NSDictionary *event) {
      [weakSelf emitXybridLoadProgress:event];
    }];
    [_adapter setErrorEmitter:^(NSDictionary *event) {
      [weakSelf emitXybridLoadError:event];
    }];
  }
  return self;
}

- (void)initialize:(NSDictionary *)options
          resolve:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject
{
  [_adapter initializeWithOptions:options resolve:resolve reject:reject];
}

- (void)setApiKey:(NSString *)apiKey
{
  [_adapter setApiKey:apiKey];
}

- (void)loadModel:(NSDictionary *)source
        requestId:(NSString *)requestId
          resolve:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject
{
  [_adapter loadModelWithSource:source requestId:requestId resolve:resolve reject:reject];
}

- (void)runModel:(double)handle
        envelope:(NSDictionary *)envelope
          config:(NSDictionary *)config
         resolve:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject
{
  [_adapter runModelWithHandle:handle envelope:envelope config:config resolve:resolve reject:reject];
}

- (void)voices:(double)handle
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject
{
  [_adapter voicesWithHandle:handle resolve:resolve reject:reject];
}

- (void)defaultVoiceId:(double)handle
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_adapter defaultVoiceIdWithHandle:handle resolve:resolve reject:reject];
}

- (void)disposeModel:(double)handle
{
  [_adapter disposeModelWithHandle:handle];
}

- (void)isModelCached:(NSString *)modelId
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  [_adapter isModelCachedWithModelId:modelId resolve:resolve reject:reject];
}

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<facebook::react::NativeXybridSpecJSI>(params);
}

@end
