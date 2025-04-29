import { Component, OnInit } from '@angular/core';
import { ICategories } from '../create-event/create-event.component';
import { ActivatedRoute, Router } from '@angular/router';
import { DbService } from '../services/db.service';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-my-event-details',
  templateUrl: './my-event-details.component.html',
  styleUrls: ['./my-event-details.component.scss']
})
export class MyEventDetailsComponent implements OnInit {
  id = 0;
  event:any;
  categories:ICategories[] = [];
  username = "";
  constructor(private route:ActivatedRoute, private dbService:DbService, private authService:AuthService, private router:Router) {
    this.username = this.authService.currentUser.username;
  }
  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      console.log(this.id = params['id']);
      this.dbService.getEventById(this.id).then(value=>{
        this.event = value;
      }).catch(e=>{
        console.log(e);
      })
  })

  }

  startEvent(){
    this.router.navigate(['/video-stream']);
  }
}
