declare module "react-native" {
  export interface EventSubscription {
    remove(): void;
  }

  export interface TurboModule {}

  export namespace CodegenTypes {
    export type EventEmitter<T> = (listener: (event: T) => void) => EventSubscription;
  }

  export const TurboModuleRegistry: {
    getEnforcing<T extends TurboModule>(name: string): T;
    get<T extends TurboModule>(name: string): T | null;
  };
}
