import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MyEventTileComponent } from './my-event-tile.component';

describe('MyEventTileComponent', () => {
  let component: MyEventTileComponent;
  let fixture: ComponentFixture<MyEventTileComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ MyEventTileComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MyEventTileComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
