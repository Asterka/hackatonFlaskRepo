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
    console.log(this.headers)
  }
  setModified(id: any){
    this.modified[id] = true;
  }
  getModified(){
    return this.modified.keys.length();
  }
  getTableData(id: any){
    return {data: this.data[id], headers: this.headers[id]};
  }
}
