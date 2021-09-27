import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class TableDataService {
  private data: any = {};
  private headers: any = {};
  private modified: any = {};
  constructor( private http: HttpClient) { }

  requestTableData(type: any){
    /*This value is hardcoded for dev purposes*/
    return this.http.get(`http://localhost:5000/table${type}`).toPromise();
  }

  setTableData(id: any, data: any, headers: any){
    this.data[id] = data;
    this.headers[id] = headers;
    //console.log(this.headers)
  }
  setModified(id: any){
    this.modified[id] = true;
  }
  getIsModified(){
    return Object.keys(this.modified).length>0;
  }
  getModified(){
    return this.modified;
  }

  getTableData(id: any){
    return {data: this.data[id], headers: this.headers[id]};
  }
  sendData(id: any){
    delete this.modified[id];
    this.http.post(`http://localhost:5000/table${id}`, this.data[id]).toPromise().then((res)=>{console.log(res)}).catch((err)=>{console.log(err)});
  }
}
