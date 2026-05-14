type Listener<T> = (event: T) => void;

type Subscription = {
  remove(): void;
};

const moduleMock: Record<string, unknown> = {};

export function __setXybridMock(mock: unknown): void {
  for (const key of Object.keys(moduleMock)) {
    delete moduleMock[key];
  }
  Object.assign(moduleMock, mock);
}

export const TurboModuleRegistry = {
  getEnforcing<T>(_name: string): T {
    return moduleMock as T;
  },
  get<T>(_name: string): T | null {
    return (moduleMock as T | undefined) ?? null;
  },
};

export function createEventEmitter<T>() {
  const listeners = new Set<Listener<T>>();
  const emitter = (listener: Listener<T>): Subscription => {
    listeners.add(listener);
    return {
      remove() {
        listeners.delete(listener);
      },
    };
  };
  emitter.emit = (event: T) => {
    for (const listener of listeners) {
      listener(event);
    }
  };
  return emitter;
}
