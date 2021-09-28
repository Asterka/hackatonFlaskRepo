import { DynamicDialogModule, DialogService } from 'primeng/dynamicdialog';
import { Input, NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { DropdownModule } from 'primeng/dropdown';
import { InputTextModule } from 'primeng/inputtext';
import { CellEditor, TableModule} from 'primeng/table';
import { Toast, ToastModule } from "primeng/toast";
import { AppComponent } from './app.component';
import { TablePageComponent } from './table-page/table-page.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CalendarModule } from 'primeng/calendar';
import {SliderModule} from 'primeng/slider';
import {DialogModule} from 'primeng/dialog';
import {MultiSelectModule} from 'primeng/multiselect';
import { ButtonModule } from 'primeng/button';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { TableDataService } from './table-data.service';
import { MessageService } from 'primeng/api';

import {PanelModule} from 'primeng/panel';

@NgModule({
  declarations: [
    AppComponent,
    TablePageComponent,
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    DynamicDialogModule,
    TableModule,
    CalendarModule,
		SliderModule,
		DialogModule,
		MultiSelectModule,
		DropdownModule,
		ButtonModule,
		ToastModule,
    InputTextModule,
    HttpClientModule,
    FormsModule,
    RouterModule,
    ToastModule,
    PanelModule
  ],
  providers: [TableDataService, MessageService, DialogService],
  bootstrap: [AppComponent]
})
export class AppModule { }
