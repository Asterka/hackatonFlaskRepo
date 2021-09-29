import { TableDataService } from './../table-data.service';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss']
})
export class GraphComponent implements OnInit {
  private isSent: boolean = false;
  public number: number = 0;
  constructor(public tableDataService: TableDataService) { }

  ngOnInit(): void {

  }

  sendTableRequest(){
    this.tableDataService.requestStrategyTable(this.number).then((data:any)=>{
      this.tableDataService.strategyTable = JSON.parse(data);
      console.log(this.tableDataService.strategyTable)
    })
  }

}
