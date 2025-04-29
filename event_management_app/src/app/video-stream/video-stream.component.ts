import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import { SocketService } from '../services/socket.service';

@Component({
  selector: 'app-video-stream',
  templateUrl: './video-stream.component.html',
  styleUrls: ['./video-stream.component.scss']
})
export class VideoStreamComponent implements AfterViewInit, OnDestroy {
  @ViewChild("videoElement") videoElement!: ElementRef;
  @ViewChild("canvasElement") canvasElement!: ElementRef;
  peopleCount: number = 0;
  streaming = false;
  videoStream!: MediaStream;

  constructor(private socketService: SocketService) {}

  ngAfterViewInit() {
    this.startCamera();
    this.socketService.onProcessedFrame((data) => {
      this.updateProcessedFrame(data);
    });
  }

  startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      this.videoElement.nativeElement.srcObject = stream;
      this.videoStream = stream;
      this.streaming = true;
      this.processFrames();
    });
  }

  processFrames() {
    const canvas = this.canvasElement.nativeElement;
    const context = canvas.getContext("2d");
    const video = this.videoElement.nativeElement;

    setInterval(() => {
      if (this.streaming) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const frameData = canvas.toDataURL("image/jpeg");
        this.socketService.sendFrame(frameData);
      }
    }, 1000);
  }

  updateProcessedFrame(data: any) {
    this.peopleCount = data.peopleCount;
    const img = new Image();
    img.src = data.frame;
    img.onload = () => {
      const canvas = this.canvasElement.nativeElement;
      const context = canvas.getContext("2d");
      context.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
  }

  ngOnDestroy() {
    this.stopCamera();
  }

  stopCamera() {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((track) => track.stop()); // Stop each track
    }
    this.streaming = false;
  }
}
