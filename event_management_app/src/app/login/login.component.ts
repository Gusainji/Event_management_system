import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AuthService } from '../services/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent {
  loginForm: FormGroup;
  constructor(private formBuilder:FormBuilder, private authService:AuthService, private router:Router){
    this.loginForm = this.formBuilder.group({
      username:['',Validators.required],
      password:['',Validators.required]
    });
  }

  postData(formData:any){
    this.authService.login(formData.value.username, formData.value.password)
    .then((value)=>{
      this.router.navigate(['/profile/my-events']);
      
    })
    .catch((error)=>{
      console.log(error);
    })
  }
}
