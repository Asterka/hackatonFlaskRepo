import { Component, OnInit } from '@angular/core';
import { MessageService } from 'primeng/api';
import { CellEditor } from 'primeng/table';
import { TableDataService } from '../table-data.service';

@Component({
  selector: 'app-table-page',
  templateUrl: './table-page.component.html',
  styleUrls: ['./table-page.component.scss']
})
export class TablePageComponent implements OnInit {

  constructor(private messageService: MessageService, private tableDataSerivce: TableDataService) { }

  ngOnInit() {
    this.tableDataSerivce.requestTableData().then((data)=>{
      console.log(data)
    })
  }

  onRowEditInit() {

  }

  onRowEditSave() {

  }

  onRowEditCancel() {
  }


}