import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';
import { BehaviorSubject, lastValueFrom, Observable, of } from 'rxjs';
import { jwtDecode } from 'jwt-decode';
import { User } from '../models/user';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private authStatus = new BehaviorSubject<boolean>(false);
  authStatus$ = this.authStatus.asObservable();
  private readonly baseUri:string = environment.apiUrl;
  private jwtToken: string = '';
  currentUser: User = {
    username: '',
    name: '',
  };

  constructor(private httpClient:HttpClient) {
    if(this.isLoggedIn()){
      let token = localStorage.getItem('event') || "";
      let tokenvalues:User = jwtDecode(token);
      this.currentUser = tokenvalues;
    }
  }

  public async login(username:string, password:string){
    return await lastValueFrom(
      this.httpClient.post<any>(this.baseUri + '/auth/login', {
        username,
        password
      })
    ).then((value) => {
      this.setToken(value.token);
      let tokenvalues:User = jwtDecode(value.token);
      this.currentUser = tokenvalues;
      return value;
    });
  }

  public async register(username:string, password:string, name:string){
    return await lastValueFrom(
      this.httpClient.post<any>(this.baseUri + '/auth/register', {
        username,
        password,
        name
      })
    ).then((value) => {
      return value;
    }
    )
  }

  setToken(token: string) {
    localStorage.setItem('event', token);
    this.authStatus.next(true);
  }
  getToken() {
    return localStorage.getItem('event');
  }
  deleteToken() {
    localStorage.removeItem('event');
    this.authStatus.next(false);
  }

  isLoggedIn(){
    const usertoken = this.getToken();
    if (usertoken != null) {
      return true;
    }
    return false;
  }
}
