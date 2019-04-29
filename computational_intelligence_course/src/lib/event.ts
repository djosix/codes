class EventManager {
  private _events = {};
  // public queue = {};

  getEvents() {
    return this._events;
  }

  getEventNames() {
    return Object.keys(this._events);
  }

  register(name: string, callback: Function) {
    this._events[name] = callback;
  }

  emit(name: string, ...args: any[]): any {
    if (!this._events[name]) {
      throw Error(`"${name}" is not registered as an event`);
    }
    return this._events[name](...args);
  }
}

export { EventManager };
export default new EventManager();
