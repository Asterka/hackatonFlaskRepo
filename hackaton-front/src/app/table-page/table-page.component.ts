import { Component, OnInit, ViewChild } from '@angular/core';
import { MessageService } from 'primeng/api';
import { CellEditor, Table } from 'primeng/table';
import { TableDataService } from '../table-data.service';

@Component({
  selector: 'app-table-page',
  templateUrl: './table-page.component.html',
  styleUrls: ['./table-page.component.scss']
})
export class TablePageComponent implements OnInit {

  constructor(private messageService: MessageService, public tableDataSerivce: TableDataService) { }
  @ViewChild('dt') table: any;

  ngOnInit() {
    this.tableDataSerivce.requestTableData().then((data: any)=>{
      this.messageService.add({'severity':'info', detail:'Данные обновлены'});
      console.log(JSON.parse(data))
      this.tableDataSerivce.setTableData(JSON.parse(data));
    })
  }

  applyFilterGlobal($event:any, stringVal: any){
      console.log($event.target.value)
      this.table.filterGlobal(($event.target as HTMLInputElement).value, 'contains');
  }
  
  onRowEditInit() {

  }

  onRowEditSave() {

  }

  onRowEditCancel() {
  }


}