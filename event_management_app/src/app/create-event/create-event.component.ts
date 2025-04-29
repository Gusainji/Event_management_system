import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { DbService } from '../services/db.service';
import { IFileHandle } from '../models/file';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-create-event',
  templateUrl: './create-event.component.html',
  styleUrls: ['./create-event.component.scss']
})
export class CreateEventComponent implements OnInit {
  eventForm:FormGroup;
  cover!: IFileHandle;
  coverSet = 0;
  categories: ICategories[] = [];

  constructor(private formBuilder:FormBuilder, private dbService:DbService,
    private sanitizer: DomSanitizer){
    this.eventForm = this.formBuilder.group({
      event_name:['',Validators.required],  
      description:['',Validators.required],
      category:['',Validators.required],
      start_date:['',Validators.required],
      end_date:['',Validators.required],
      start_time:['',Validators.required],
      duration:['',Validators.required],
      location:['',Validators.required],
      price:['',Validators.required],
      is_online:['true', Validators.required],
      is_free:['true', Validators.required]
    })
  }

  ngOnInit(): void {
    this.dbService.getCategories().then(value=>{
      console.log(value);
      this.categories = value;
      this.categories.sort((a,b) => a.name > b.name ? 1 : -1)
    }).catch(e=>{
      console.log(e);
    })
  }

  onCoverSelect(event:any){
    if (event.target.files) {
      const photo = event.target.files[0];
      const fileHandle: IFileHandle = {
        file: photo,
        url: this.sanitizer.bypassSecurityTrustUrl(
          window.URL.createObjectURL(photo)
        ),
      };
      this.cover = fileHandle;
      this.coverSet = 1;
    }
  }

  postData(formData:any){
    let start_time = new Date(formData.value.start_date+"T"+formData.value.start_time).getTime();
    let end_date = new Date(formData.value.end_date).getTime();
    let duration = formData.value.duration;
    this.dbService.createEvent(
      formData.value.event_name,
      formData.value.description,
      formData.value.category,
      start_time,
      duration,
      end_date,
      formData.value.is_online,
      formData.value.location,
      formData.value.is_free,
      formData.value.price,
      this.cover.file
    ).then(value=>{
      console.log(value);
    }).catch(e=>{
      console.log(e);
    })
  }
  setStartDate(event:any){
    console.log(event);
  }
}

export interface ICategories{
  id:number;
  name:string;
}
