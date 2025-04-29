import { Injectable } from '@angular/core';
import { io } from "socket.io-client";
@Injectable({
  providedIn: 'root'
})
export class SocketService {

  constructor() { }
  private socket = io("http://127.0.0.1:5000");

  sendFrame(frame: string) {
    this.socket.emit("frame", frame);
  }

  onProcessedFrame(callback: (data: any) => void) {
    this.socket.on("processedFrame", callback);
  }
}
