import { Component, OnInit } from '@angular/core';
import { MessageService } from 'primeng/api';
import { CellEditor } from 'primeng/table';

@Component({
  selector: 'app-table-page',
  templateUrl: './table-page.component.html',
  styleUrls: ['./table-page.component.scss']
})
export class TablePageComponent implements OnInit {

  constructor(private messageService: MessageService) { }

  ngOnInit() {

  }

  onRowEditInit() {

  }

  onRowEditSave() {

  }

  onRowEditCancel() {
  }


}