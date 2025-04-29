import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { DbService } from '../services/db.service';
import { AuthService } from '../services/auth.service';
import { ICategories } from '../create-event/create-event.component';

@Component({
  selector: 'app-event',
  templateUrl: './event.component.html',
  styleUrls: ['./event.component.scss']
})
export class EventComponent implements OnInit {
  id = 0;
  event:any;
  isEnrolled = false;
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
        if(this.authService.isLoggedIn()){
          this.dbService.isEnrolled(this.id).then(value=>{
            this.isEnrolled = value;
          }
          ).catch(e=>{
            console.log(e);
          }
          )
        }
      }).catch(e=>{
        console.log(e);
      })
  })

  }

  enroll(){
    if(this.authService.isLoggedIn()){
      this.dbService.enrollEvent(this.id).then(value=>{
        console.log(value);
        this.isEnrolled = true;
      }).catch(e=>{
        console.log(e);
      })
    }
    else{
      this.router.navigate(['/login']);
    }
  }

  unenroll(){
    this.dbService.unenrollEvent(this.id).then(value=>{
      console.log(value);
      this.isEnrolled = false;
    }).catch(e=>{
      console.log(e);
    })
  }
}
