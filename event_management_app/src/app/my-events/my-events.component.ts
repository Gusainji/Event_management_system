import { Component, OnInit } from '@angular/core';
import { DbService } from '../services/db.service';

@Component({
  selector: 'app-my-events',
  templateUrl: './my-events.component.html',
  styleUrls: ['./my-events.component.scss']
})
export class MyEventsComponent implements OnInit {
  events: any = [];
  constructor( private dbService: DbService) { }
  ngOnInit(): void {
    this.dbService.getMyEvents().then(value => {
      this.events = value;
      console.log(this.events);
    }).catch(e => {
      console.log(e);
    })
  }

}
