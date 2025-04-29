import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { lastValueFrom } from 'rxjs';
import { environment } from 'src/environments/environment';
import { AuthService } from './auth.service';

@Injectable({
  providedIn: 'root'
})
export class DbService {
  jwtToken: string = '';  
  private readonly baseUri:string = environment.apiUrl;
  constructor(private httpClient:HttpClient, private authService:AuthService) { }


  public async getCategories(){
    return await lastValueFrom(
      this.httpClient.get<any>(this.baseUri + '/categories/all')
    ).then(value=>{
      return value;
    })
  }



  public async createEvent(event_name:string, description:string, category:string, start_time:number, duration:string, end_date:number, location_type:string, location:string, freeornot:string, price:number, cover:any){
    const header = this.createHeader();
    const formData = new FormData();
    formData.append('event_name', event_name);
    formData.append('description', description);
    formData.append('category', category);
    formData.append('start_time', start_time.toString()); 
    formData.append('duration', duration);
    formData.append('end_date', end_date.toString());
    formData.append('is_online', location_type);
    formData.append('location', location);
    formData.append('is_free', freeornot);
    formData.append('price', price.toString());
    formData.append('cover', cover);
    return await lastValueFrom(
      this.httpClient.post<any>(this.baseUri + '/events/create', formData, { headers: header })
    ).then(value=>{
      return value;
    })
  }

  public async getEvents(){
    return await lastValueFrom(
      this.httpClient.get<any>(this.baseUri + '/events/all')
    ).then(value=>{
      return value;
    })
  }

  public async getMyEvents(){
    const header = this.createHeader();
    return await lastValueFrom(
      this.httpClient.get<any>(this.baseUri + '/events/myevent', { headers: header })
    ).then(value=>{
      return value;
    })
  }

  public async getEventById(id:number){
    return await lastValueFrom(
      this.httpClient.get<any>(this.baseUri + '/events/id/' + id)
    ).then(value=>{
      return value;
    })
  }

  public async enrollEvent(id:number){
    const header = this.createHeader();
    return await lastValueFrom(
      this.httpClient.post<any>(this.baseUri + '/events/enroll/' + id, {}, { headers: header })
    ).then(value=>{
      return value;
    })
  }

  public async getMyTickets(){
    const header = this.createHeader();
    return await lastValueFrom(
      this.httpClient.get<any>(this.baseUri + '/events/mytickets', { headers: header })
    ).then(value=>{
      return value;
    })
  }

  public async isEnrolled(id:number){
    const header = this.createHeader();
    return await lastValueFrom(
      this.httpClient.get<any>(this.baseUri + '/events/isenrolled/' + id, { headers: header })
    ).then(value=>{
      return value;
    })
  }

  public async unenrollEvent(id:number){
    const header = this.createHeader();
    return await lastValueFrom(
      this.httpClient.post<any>(this.baseUri + '/events/unenroll/' + id, {}, { headers: header })
    ).then(value=>{
      return value;
    })
  }

  private createHeader(){
    if(this.authService.isLoggedIn())
      this.jwtToken = "bearer " + this.authService.getToken() || "";
    let header = new HttpHeaders().set(
      "Authorization",
       this.jwtToken
    );
    return header;
   }
}
