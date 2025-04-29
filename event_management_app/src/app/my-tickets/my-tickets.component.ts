import { Component } from '@angular/core';
import { DbService } from '../services/db.service';

@Component({
  selector: 'app-my-tickets',
  templateUrl: './my-tickets.component.html',
  styleUrls: ['./my-tickets.component.scss']
})
export class MyTicketsComponent {
  constructor(private dbService: DbService) { }
  tickets: any[] = [];
  ngOnInit(): void {
    this.dbService.getMyTickets().then(value => {
      this.tickets = value;
    }).catch(e => {
      console.log(e);
    })
  }
}
