import { Component, OnInit, ViewChild } from '@angular/core';
import { MessageService } from 'primeng/api';
import { CellEditor, Table } from 'primeng/table';
import { TableDataService } from '../table-data.service';
import {TableModule} from 'primeng/table';

@Component({
  selector: 'app-table-page',
  templateUrl: './table-page.component.html',
  styleUrls: ['./table-page.component.scss']
})
export class TablePageComponent implements OnInit {

  constructor(private messageService: MessageService, public tableDataSerivce: TableDataService) { }
  @ViewChild('dt') table: any;
  public editing = false;
  ngOnInit() {
    this.tableDataSerivce.requestTableData().then((data: any)=>{
      this.messageService.add({'severity':'info', detail:'Данные обновлены'});
      data = <Array<any>>JSON.parse(data);
      let headers = data[0];
      this.tableDataSerivce.setTableData(data.slice(1), headers);
    })
  }

  applyFilterGlobal($event:any, stringVal: any){
      console.log($event.target.value, this.table)
      this.table.filterGlobal(($event.target as HTMLInputElement).value, 'contains');
  }
  
  onRowEditInit() {
    console.log('Focus');
  }

  onValueUpdate(event:any, row: any, id:any) {
    console.log('here');
    row[id] = event.target.value
    console.log(row[id])
  }

  onRowEditCancel() {
  }


}