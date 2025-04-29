import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-tags',
  templateUrl: './tags.component.html',
  styleUrls: ['./tags.component.scss']
})
export class TagsComponent {
 @Input() text = "default";
 @Input() isFree = true;
 @Input() isOnline = true;
 @Input() price = 0;
 constructor() { }
}
