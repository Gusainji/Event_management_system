import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './header/header.component';
import { FooterComponent } from './footer/footer.component';
import { HomeComponent } from './home/home.component';
import { LoginComponent } from './login/login.component';
import { RegisterComponent } from './register/register.component';
import { TitleComponent } from './title/title.component';
import { ProfileComponent } from './profile/profile.component';
import { CreateEventComponent } from './create-event/create-event.component';
import { FeedTileComponent } from './feed-tile/feed-tile.component';
import { TagsComponent } from './tags/tags.component';
import { NgxPaginationModule } from 'ngx-pagination';
import { MyEventsComponent } from './my-events/my-events.component';
import { MyTicketsComponent } from './my-tickets/my-tickets.component';
import { EventComponent } from './event/event.component';
import { MyEventTileComponent } from './my-event-tile/my-event-tile.component';
import { MyEventDetailsComponent } from './my-event-details/my-event-details.component';
import { VideoStreamComponent } from './video-stream/video-stream.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    FooterComponent,
    HomeComponent,
    LoginComponent,
    RegisterComponent,
    TitleComponent,
    ProfileComponent,
    CreateEventComponent,
    FeedTileComponent,
    TagsComponent,
    MyEventsComponent,
    MyTicketsComponent,
    EventComponent,
    MyEventTileComponent,
    MyEventDetailsComponent,
    VideoStreamComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    NgxPaginationModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
