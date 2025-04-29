import { Component, Input } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-my-event-tile',
  templateUrl: './my-event-tile.component.html',
  styleUrls: ['./my-event-tile.component.scss']
})
export class MyEventTileComponent {
  constructor(private router: Router) { }
  @Input() eventId = 1;
  @Input() eventName = "JS Mastery";
  @Input() eventOrganizer = "Sachin Gusian";
  @Input() eventPic = "demo.jpg";
  @Input() date = new Date().getTime();
  @Input() isFree = true;
  @Input() price = 0;
  @Input() isOnline = true;

  goToEvent(){
    this.router.navigate(['/profile/my-events/details'], { queryParams: { id: this.eventId } });
  }
}
