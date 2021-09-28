import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { TablePageComponent } from './table-page/table-page.component';

const routes: Routes = [
  {path:"table1", component:TablePageComponent},
  {path:"table2", component:TablePageComponent},
  {path:"table3", component:TablePageComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
