use std::fs::read_to_string;
use ndarray::*;
use ndarray_linalg::*;
use std::env;
use std::time;

//define type name
type Array2df64 = ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>;
type Array1df64 = ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>;
type Vet = Vec<f64>;
type Swap = Vec<(usize,usize)>;

//if x < EPS , consider x < 0
const EPS : f64 = -0.000_000_000_000_1;

//define Struct, Enum
#[derive(Debug)]
struct Csvdata{
    m : u64,
    n : u64,
    a : Vet,
    b : Vet,
    c : Vet,
}

#[derive(Debug)]
struct Model{
    ab : Array2df64,
    an : Array2df64,
    cb : Array1df64,
    cn : Array1df64,
    xb : Array1df64,
    xn : Array1df64,
    swap_data : Vec<(usize,usize)>,
    solve_status : Solvestatus,
}

#[derive(PartialEq)]
#[derive(Debug)]
enum Solvestatus{
    Unbounded,
    Solved,
    Running,
}

enum Readstatus{
    MN,
    A(i32), 
    B(i32),
    C(i32),
    END,
}

#[derive(Clone)]
#[derive(Copy)]
enum BN{
    B,
    N,
}


fn main(){
    env::set_var("RUST_BACKTRACE", "1");
    //--------read data--------
    println!("Reading data");

    let path = "./src/simplex_method_csvs/feasible1.csv";

    let csvdata : Csvdata = read_csv(path);
    let m: usize = csvdata.m as usize;
    let n: usize = csvdata.n as usize;


    let mut a_mat : Array2df64 = Array::from_shape_vec((m, n) , csvdata.a).unwrap();
    let mut b_vec : Array1df64 = Array::from(csvdata.b);
    let c_vec : Array1df64 = Array::from(csvdata.c);

    println!("Complete reading data");
    //--------read data end--------
    let now = time::Instant::now();

    //execute equivalence transformation
    //b should > 0 (all component)
    convert_b_non_negative(&mut a_mat, &mut b_vec);

    //--------step 0--------
    println!("step 0");
    let ab : Array2df64 = Array::eye(m);
    let an = a_mat.clone();
    
    let cb : Array1df64 = Array::ones(m);
    let cn : Array1df64 = Array::zeros(n);

    let xb = b_vec.clone();
    let xn : Array1df64 = Array::zeros(n);

    let swap_data : Swap = vec![];
    let solve_status = Solvestatus::Running;

    let mut step0_model = Model{ab,an,cb,cn,xb,xn,swap_data,solve_status};

    solve(&mut step0_model);

    println!("step 0 : {:?}",step0_model.solve_status);
    //--------step 0 end--------

    //--------step 1~4--------
    println!("step 1~4");
    let ( mut step1_4_model,bn_n) = convert_step0_model_2_step_1_4_model(&step0_model,c_vec);
    
    solve(&mut step1_4_model);

    println!("step 1~4 : {:?}",step1_4_model.solve_status);
    //--------step 1~4 end--------

    //--------print answer--------
    let answerx = restore_x(&mut step1_4_model.xb.clone(), &mut step1_4_model.xn.clone(),step1_4_model.swap_data,bn_n,step0_model.swap_data);

    //output elapsed time
    println!("elapsed time : {:?}", now.elapsed());

    println!("answer : {:?}",answerx);

    //check |Ax - b| = 0
    let norm_ax_b = (&a_mat.dot(&answerx) - &b_vec).norm_l2();
    println!("|Ax - b| = {}",norm_ax_b);
    //--------print answer end--------
}

fn solve(model : &mut Model){
    let mut  count = 0;
    while model.solve_status == Solvestatus::Running{
        println!("count : {}, cTx : {:?}",count,&model.cb.t().dot(&model.xb) + &model.cn.t().dot(&model.xn));

        let f_ab = model.ab.factorize().unwrap();
        let f_ab_t = model.ab.t().factorize().unwrap();

        /*
        https://github.com/rust-ndarray/ndarray-linalg/blob/master/lax/src/lib.rs
        //------------------------------------------------------------------------
        Error
        ------
        - if the matrix is singular
        - On this case, `return_code` in [Error::LapackComputationalFailure] means
        `return_code`-th diagonal element of $U$ becomes zero.

        //------------------------------------------------------------------------

        run result (On mac , feasible4)
        //------------------------------------------------------------------------
        Reading data
        Complete reading data
        step 0
        count : 0, cTx : 43.26243565382619
        count : 1, cTx : 42.85724141744498
        count : 2, cTx : 42.7342155492866
        count : 3, cTx : 41.89985374046676
        count : 4, cTx : 37.62312550658253
        count : 5, cTx : 37.57968449838777
        count : 6, cTx : 37.419129336524044
        thread 'main' panicked at 'called `Result::unwrap()` on an `Err` 
        value: Lapack(LapackComputationalFailure { return_code: 1 })', src/optimization/bin/simplex_method.rs:129:41
        //------------------------------------------------------------------------

        */

        //solve AbT pi = Cb
        let pi = f_ab_t.solve(&model.cb).unwrap();

        //calculate Cn - AnTpi
        let reduced_cost = &model.cn - &model.an.t().dot(&pi);

        //some components are negative -> Return minimum suffix, 
        //no component is negative -> return None
        let first_negative = first_negative(&reduced_cost);

        match first_negative {
            None => {
                model.solve_status = Solvestatus::Solved;
                break;
            },
            _ => {}
        }

        let k_of_n = first_negative.unwrap();

        //step3
        //solbe Aby = (An)_{row k}
        let y = f_ab.solve(&model.an.column(k_of_n)).unwrap();

        let delta = &model.xb / &y;

        let min_non_negative = min_non_negative(&y , &delta);

        match min_non_negative {
            None => {
                model.solve_status = Solvestatus::Unbounded;
                break;
            },
            _ => {}
        }

        let i_of_b = min_non_negative.unwrap();

        update(model,y,k_of_n,i_of_b);
        count += 1;
    }
}

fn update(model : &mut Model, y : Array1df64, k_of_n : usize, i_of_b : usize){
    //update & swap x
    let delta_min = model.xb[i_of_b] / y[i_of_b];

    model.xb = &model.xb - &(y * (delta_min));
    model.xb.slice_mut(s![i_of_b]).fill(delta_min);

    //swap a
    let ab_i = model.ab.column(i_of_b).to_owned();
    let an_k = model.an.column(k_of_n).to_owned();

    for j in 0..model.ab.nrows(){
        model.ab.column_mut(i_of_b)[j] = an_k[j];
    }
    for j in 0..model.an.nrows(){
        model.an.column_mut(k_of_n)[j] = ab_i[j];
    }
    
    //swap c
    swap_vec_bn(&mut model.cb, &mut model.cn, i_of_b, k_of_n);
    //update swap_data
    model.swap_data.push((i_of_b,k_of_n));
}

fn convert_b_non_negative(a : &mut Array2df64, b : &mut Array1df64){

    for i in 0..b.len(){
        if b[i] < 0.{
            let orgin = b[i]*(-1.);
            b.slice_mut(s![i]).fill(orgin);

            for j in 0..a.ncols(){
                a.row_mut(i)[j] = a.row(i)[j] *(-1.);
            }
            
        }
    }
}

fn first_negative(x : &Array1df64) -> Option<usize>{
    let mut counter : usize = 0;

    let mut answer = None;

    while counter < x.len(){
        if x[counter] < EPS{
            answer = Some(counter);
            break;
        }
        counter += 1;
    }

    answer
}

fn min_non_negative(y : &Array1df64,delta : &Array1df64) -> Option<usize>{
    let mut answer = None;

    for counter in 0..y.len(){
        if y[counter] >= 0.{
            match answer{
                None => {
                    answer = Some(counter);
                },
                Some(i) => {
                    if delta[i] > delta[counter]{
                        answer = Some(counter);
                    }
                },
            }
        }
    }

    answer
}

fn convert_step0_model_2_step_1_4_model(step0_model : &Model , c_vec : Array1df64) -> (Model,Vec<BN>){
    let m = step0_model.ab.ncols();
    let n = step0_model.an.ncols();

    let mut bn_b = vec![BN::B; m];
    let mut bn_n = vec![BN::N; n];

    let mut cb : Array1df64 = Array::zeros(m);
    let mut cn_raw = c_vec; 

    //swap bn and c

    for index in 0..step0_model.swap_data.len(){
        let (i_of_b,k_of_n) = step0_model.swap_data[index];
        let b_n_i = bn_b[i_of_b];
        let b_n_k = bn_n[k_of_n];
        bn_b[i_of_b] = b_n_k;
        bn_n[k_of_n] = b_n_i;

        swap_vec_bn(&mut cb, &mut cn_raw, i_of_b, k_of_n);
    }

    let num_b_in_swapped_bn_b : i8 = bn_b.into_iter().map(|bn| match bn{BN::B => 1,BN::N => 0}).sum();

    //check cTx == 0; otherwise infeasible
    if num_b_in_swapped_bn_b > 0{
        panic!("infeasible");
    }

    let ab = step0_model.ab.clone();
    let mut an : Array2df64 = Array::zeros((m,n-m));
    //already difine cb
    let mut cn : Array1df64 = Array::zeros(n-m);
    let xb = step0_model.xb.clone();
    let xn : Array1df64 = Array::zeros(n-m);

    let swap_data : Swap = vec![];
    let solve_status = Solvestatus::Running;

    //update an, cn
    let mut count : usize = 0;
    for index in 0..bn_n.len(){
        match bn_n[index]{
            BN::B => {},
            BN::N => {
                
                for jndex in 0..m{
                    an.column_mut(count)[jndex] = step0_model.an.column(index)[jndex];
                }

                let cn_raw_index = cn_raw[index];
                cn.slice_mut(s![count]).fill(cn_raw_index);
                count += 1;
            }
        }
    }

    (Model{ab,an,cb,cn,xb,xn,swap_data,solve_status},bn_n)


}

fn restore_x(xb : &mut Array1df64, xn : &mut Array1df64, step1_4_swap_data : Swap, bn_n : Vec<BN> ,step0_swap_data : Swap) -> Array1df64{

    //--------restore x by following procedure--------

    //xb,xn -> xb,xn (use step1_4_swap_data)
    //xb,xn -> xb,new_xn (use bn_n)
    //xb,new_xn -> xb,new_xn (use step0_swap_data)

    //------------------------------------------------

    //xb,xn -> xb,xn (use step1_4_swap_data)
    for i_k in step1_4_swap_data.into_iter().rev(){
        swap_vec_bn(xb, xn, i_k.0, i_k.1);
    }

    //xb,xn -> xb,new_xn (use bn_n)
    let mut new_xn_vec = Vec::new();

    let mut counter : usize = 0;
    for bn in bn_n{
        match bn {
            BN::B =>{
                new_xn_vec.push(0.);
            }
            BN::N =>{
                new_xn_vec.push(xn[counter]);
                counter += 1;
            }
        } 
    }

    let mut new_xn = Array::from(new_xn_vec);

    //xb,new_xn -> xb,new_xn (use step0_swap_data)
    for i_k in step0_swap_data.into_iter().rev(){
        swap_vec_bn(xb,  &mut new_xn, i_k.0, i_k.1);
    }

    new_xn
}

fn swap_vec_bn(vec_b : &mut Array1df64, vec_n : &mut Array1df64, index : usize, kndex : usize){
    
    let vec_b_index = vec_b[index];
    let vec_n_kndex = vec_n[kndex];
    vec_b.slice_mut(s![index]).fill(vec_n_kndex);
    vec_n.slice_mut(s![kndex]).fill(vec_b_index);
}

fn read_csv(path : &str) -> Csvdata{

    let open = read_to_string(path).unwrap();

    convert_string_2_condtion(open)
}

fn convert_string_2_condtion(string : String) -> Csvdata{
    let lines = string.lines();

    //println!("{}",string);

    let mut m: u64 = 0;
    let mut n: u64 = 0;

    let mut a : Vet = Vec::new();
    let mut b : Vet = Vec::new();
    let mut c : Vet = Vec::new();

    let mut status = Readstatus::MN;

    for line in lines{
        let divided : Vec<&str> = line.split(',').collect();

        match status{
            Readstatus::MN => {
                m = divided[0].parse().expect("not a number");
                n = divided[1].parse().expect("not a number");

                status = Readstatus::A(-1);

            },
            Readstatus::A(-1) => {
                status = Readstatus::A(0);
            },
            Readstatus::A(i) => {

                for num_string in divided{
                    let num : f64 = num_string.parse().expect("not a number");
                    a.push(num);
                }

                if i == m as i32 - 1{

                    status = Readstatus::B(-1);
                }
                else{
                    status = Readstatus::A(i+1);
                }

            },
            Readstatus::B(-1) => {
                status = Readstatus::B(0);
            },
            Readstatus::B(_) => {
                for num_string in divided{
                    let num : f64 = num_string.parse().expect("not a number");
                    
                    b.push(num);
                }
                status = Readstatus::C(-1);

            },
            Readstatus::C(-1) => {
                status = Readstatus::C(0);
            },
            Readstatus::C(_) => {
                for num_string in divided{
                    let num : f64 = num_string.parse().expect("not a number");

                    c.push(num);
                }
                status = Readstatus::END;

            },
            Readstatus::END => {}
            
        }
    }

    Csvdata{
        m,n,a,b,c
    }
}