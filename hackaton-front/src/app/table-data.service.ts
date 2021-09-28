import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MessageService } from 'primeng/api';

@Injectable({
  providedIn: 'root',
})
export class TableDataService {
  private data: any = {};
  private headers: any = {};
  private modified: any = {};
  private shouldUpdate: any = {};
  constructor(private http: HttpClient, private messageService: MessageService) {}

  requestTableData(type: any) {
    /*This value is hardcoded for dev purposes*/
    return this.http.get(`http://localhost:12345/table${type}`).toPromise();
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
  sendData(id: any) {
    delete this.modified[id];
    this.http
      .post(`http://localhost:12345/table${id}`, JSON.stringify(this.data[id]))
      .toPromise()
      .then((res:any) => {
        console.log('here')
        this.requestTableData(id).then((data: any)=>{
          this.messageService.add({'severity':'info', detail:'Данные обновлены'});
          data = <Array<any>>JSON.parse(data);
          let headers = data[0];

          /* Save the parsed data under its id, split headers */
          this.setTableData(Number.parseInt(id), data.slice(1), headers);
        });
      })
      .catch((err) => {
        this.messageService.add({'severity':'error', detail:'Произошла ошибка при обработке запроса'});
        console.log(err);
      });
  }
}
