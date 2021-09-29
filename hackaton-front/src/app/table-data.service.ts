import { GraphComponent } from './graph/graph.component';
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MessageService } from 'primeng/api';
import { DialogService } from 'primeng/dynamicdialog';

@Injectable({
  providedIn: 'root',
})
export class TableDataService {
  private data: any = {};
  private headers: any = {};
  private modified: any = {};
  private shouldUpdate: any = {};
  public strategyTable: any = [];
  public plotActive: boolean = false;
  constructor(private http: HttpClient, private messageService: MessageService, public dialogService: DialogService) {}

  requestTableData(type: any) {
    /*This value is hardcoded for dev purposes*/
    return this.http.get(`http://3.22.224.152:12345/table${type}`).toPromise();
  }

  requestStrategyTable(number: number) {
    return this.http.post(`http://3.22.224.152:12345/table4`, {'number':number}).toPromise();
  }

  setTableData(id: any, data: any, headers: any) {
    this.data[id] = data;
    this.headers[id] = headers;
    //console.log(this.headers)
  }
  setModified(id: any) {
    this.modified[id] = true;
  }
  getIsModified() {
    return Object.keys(this.modified).length > 0;
  }
  getModified() {
    return this.modified;
  }
  setShouldUpdate(id: any, bool: boolean){
    this.shouldUpdate[id] = bool;
  }
  getTableData(id: any) {
    return { data: this.data[id], headers: this.headers[id] };
  }
  getShouldUpdate(){
    return this.shouldUpdate;
  }
  getPlot(){
    return this.http.get(`http://3.22.224.152:12345/plot`).toPromise().then(()=>{
      this.dialogService.open(GraphComponent, {
        header: 'Оптимизация риска',
        width: '70%'
      });
      this.plotActive = true;
    })
  }
  sendData(id: any) {
    delete this.modified[id];
    this.http
      .post(`http://3.22.224.152:12345/table${id}`, JSON.stringify(this.data[id]))
      .toPromise()
      .then((res:any) => {

        this.requestTableData(id).then((data: any)=>{
          this.messageService.add({'severity':'info', detail:'Данные обновлены'});
          data = <Array<any>>JSON.parse(data);

          let headers = data[0];

          /* Save the parsed data under its id, split headers */
          this.setTableData(Number.parseInt(id), data.slice(1), headers);
        });
      })
      .catch((err) => {
        this.messageService.add({'severity':'error', detail:'Ошибка обновления данных'});
      });
  }
}
