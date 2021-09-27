import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class TableDataService {
  private data: any = [];
  private headers: any = [];
  constructor( private http: HttpClient) { }

  requestTableData(){
    /*This value is hardcoded for dev purposes*/
    return this.http.get('http://localhost:5000/table1').toPromise();
  }

  setTableData(data: any, headers: any){
    this.data = data;
    this.headers = headers;
    console.log(this.headers)
  }
  getTableData(){
    return {data: this.data, headers: this.headers};
  }
}
