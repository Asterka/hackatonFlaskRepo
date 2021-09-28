import { Component } from '@angular/core';
import { TableDataService } from './table-data.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  constructor(public tableDataService: TableDataService){
  }

  update(){
    let modified: any = this.tableDataService.getModified()
    Object.keys(modified).forEach((id:any) => {
      console.log(id);
      this.tableDataService.sendData(id);
    });
  }
  isModified(){
    return this.tableDataService.getIsModified();
  }
  title = 'hackaton-front';
}
