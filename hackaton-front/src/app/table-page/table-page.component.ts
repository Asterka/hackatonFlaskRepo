import { Component, Input, OnInit, ViewChild } from '@angular/core';
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
  @Input()
  public type: string = '';

  ngOnInit() {
    this.tableDataSerivce.requestTableData(this.type).then((data: any)=>{
      this.messageService.add({'severity':'info', detail:'Данные обновлены'});
      data = <Array<any>>JSON.parse(data);
      let headers = data[0];
      
      /* Save the parsed data under its id, split headers */
      this.tableDataSerivce.setTableData(Number.parseInt(this.type), data.slice(1), headers);
    })
  }

  applyFilterGlobal($event:any, stringVal: any){
      //console.log($event.target.value, this.table)
      this.table.filterGlobal(($event.target as HTMLInputElement).value, 'contains');
  }
  
  onRowEditInit() {
    console.log('Focus');
  }

  onValueUpdate(event:any, row: any, id:any) {
    row[id] = event.target.value;
    this.tableDataSerivce.setModified(this.type);
  }

  onRowEditCancel() {
  }


}