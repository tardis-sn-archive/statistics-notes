(* ::Package:: *)

BeginPackage["MyBinarySearch`"]
binSearchMin::usage = 
"binSearchMin[list, key] returns the index of the smallest element of the sorted list that is larger or equal to key.
0 is returned if no such element exists in the list. 
For consecutive identical values, the index of the first such value is returned."
binSearchMax::usage = 
"binSearchMax[list, key] returns the index of the largest element of the sorted list that is less than or equal to key.
0 is returned if no such element exists in the list. 
For consecutive identical values, the index of the last such value is returned."

Begin["`Private`"]


(* ::Title:: *)
(*Follow http://googleresearch.blogspot.de/2006/06/extra-extra-read-all-about-it-nearly.html*)


(* ::Section:: *)
(*Return first element >= key*)


(* ::Input:: *)
(*binSearchMin[list_List, key_]:=Module[{low=1,high=Length[list],mid,midVal},*)
(*(* first element is the right one *)*)
(*If[First[list]>=key,Return[1]];*)
(*(* key larger than everything in list *)*)
(*If[Last[list]<key,Return[0]];*)
(**)
(*(* else it is inside the list *)*)
(*While[low< high-1,*)
(*mid=Quotient[low+high,2];*)
(*midVal = list[[mid]];*)
(*(*Print[{mid,midVal}];*)*)
(*If[midVal>= key,high=mid,*)
(*If[midVal<key,low=mid];*)
(*];*)
(*];*)
(*high*)
(*]*)


(* ::Section:: *)
(*Return last element <= key*)


(* ::Input:: *)
(*binSearchMax[list_List, key_]:=Module[{low=1,high=Length[list],mid,midVal},*)
(*(* first element too large => element cannot be found *)*)
(*If[First[list]>key,Return[0]];*)
(*(* key larger than everything in list *)*)
(*If[Last[list]<= key,Return[high]];*)
(**)
(*(* else it is inside the list *)*)
(*While[low< high-1,*)
(*mid=Quotient[low+high,2];*)
(*midVal = list[[mid]];*)
(*(*Print[{mid,midVal}];*)*)
(*If[midVal<=  key,low=mid,*)
(*If[midVal> key,high=mid];*)
(*];*)
(*];*)
(*low*)
(*]*)
(*data={1,3,6,7,9,9,9,9,9,10};*)
(*Length[data]*)
(*a=binSearchMax[data,9]*)
(*data[[a]]*)
(**)


(* ::Section:: *)
(*Testing*)


(* ::Input:: *)
(*On[Assert];*)
(*data={1,3,6,7,9,9,9,9,10};*)


(* ::Text:: *)
(*Need Apply to index arg as #1, #2...*)


(* ::Input:: *)
(*Map[#[[1]]+#[[2]]&,{{x,b},{c,d}},{1}]*)
(*Map[Apply[#1+#2&],{{x,b},{c,d}},{1}]*)
(**)


(* ::Input:: *)
(*(* test for (key, ret. value) pairs *)*)


(* ::Input:: *)
(*Map[Apply[Assert[binSearchMax[data,#1]==#2]&],{{0.1,0},{1,1},{2.2,1}, {7,4},{9,8}, {9.2,8},{11,9}}]*)


(* ::Input:: *)
(*Apply[Assert[binSearchMin[data,#1]==#2]&]/@{{0.1,1},{1,1},{2.2,2}, {7,4},{9,5}, {9.2,9},{11,0}}*)


End[]
EndPackage[]
