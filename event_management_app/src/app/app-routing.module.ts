import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { LoginComponent } from './login/login.component';
import { RegisterComponent } from './register/register.component';
import { ProfileComponent } from './profile/profile.component';
import { CreateEventComponent } from './create-event/create-event.component';
import { MyEventsComponent } from './my-events/my-events.component';
import { MyTicketsComponent } from './my-tickets/my-tickets.component';
import { EventComponent } from './event/event.component';
import { AuthGuard } from './guards/auth.guard';
import { MyEventDetailsComponent } from './my-event-details/my-event-details.component';
import { VideoStreamComponent } from './video-stream/video-stream.component';

const routes: Routes = [
  { path: 'home', component:HomeComponent, },
  { path: 'create-event', component: CreateEventComponent, canActivate:[AuthGuard] },
  { path: 'profile', component: ProfileComponent, children:[
    { path: 'my-events', component: MyEventsComponent, children:[
      { path: 'details', component: MyEventDetailsComponent }
    ] },
    { path: 'my-tickets', component: MyTicketsComponent }
  ], canActivate:[AuthGuard] },
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  { path: 'event', component:EventComponent},
  { path:'video-stream', component:VideoStreamComponent },
  { path: '**', redirectTo:'home'}
];

@NgModule({
  imports: [RouterModule.forRoot(routes, { useHash: true })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
