import { Component, OnInit } from '@angular/core';
import { DbService } from '../services/db.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  events: any = [];
  p: number = 1;
  constructor(private dbService:DbService) { }  

  ngOnInit(): void {
    this.dbService.getEvents().then(value=>{    
      this.events = value;
      console.log(this.events);
    }).catch(e=>{     
      console.log(e);
    })
  }
    

}
