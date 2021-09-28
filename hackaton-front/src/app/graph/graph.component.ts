import { TableDataService } from './../table-data.service';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss']
})
export class GraphComponent implements OnInit {

  constructor(public tableDataService: TableDataService) { }

  ngOnInit(): void {
    this.tableDataService.requestStrategyTable().then((data:any)=>{
      this.tableDataService.strategyTable = data;
    })
  }

}
