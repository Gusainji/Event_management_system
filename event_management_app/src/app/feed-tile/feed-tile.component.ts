import { query } from '@angular/animations';
import { Component, Input, OnChanges, OnInit, SimpleChanges } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-feed-tile',
  templateUrl: './feed-tile.component.html',
  styleUrls: ['./feed-tile.component.scss']
})
export class FeedTileComponent implements OnChanges, OnInit {
  @Input() eventId = 1;
  @Input() eventName = "JS Mastery";
  @Input() eventOrganizer = "Sachin Gusian";
  @Input() eventPic = "demo.jpg";
  @Input() date = new Date().getTime();
  @Input() isFree = true;
  @Input() price = 0;
  @Input() isOnline = true;

  firstText = "Free";
  secondText = "Online";

  constructor(private router:Router) {}
  ngOnChanges(changes: SimpleChanges): void {
    if (!this.isFree) {
      this.firstText = '₹'+this.price.toString();
    }
    if (!this.isOnline) {
      this.secondText = "Offline";
    }
  }

  ngOnInit(): void {
    

  }
  goToEvent() {
    this.router.navigate(['/event'], { queryParams: { id: this.eventId } });  
  }

}
