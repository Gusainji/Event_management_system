import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MyEventDetailsComponent } from './my-event-details.component';

describe('MyEventDetailsComponent', () => {
  let component: MyEventDetailsComponent;
  let fixture: ComponentFixture<MyEventDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ MyEventDetailsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MyEventDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
