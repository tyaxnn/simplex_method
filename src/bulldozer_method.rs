use std::fs::read_to_string;
use ndarray::*;
use ndarray_linalg::*;
use std::env;
use std::time;

mod ncm;
use ncm::allncm::ncm_enum;

//type name
type Array2df64 = ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>;
type Array1df64 = ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>;

type Vet = Vec<f64>;

const EPS : f64 = -0.000_000_000_000_1;

#[derive(Debug)]
struct Csvdata{
    m : u64,
    n : u64,
    a : Vet,
    b : Vet,
    c : Vet,
}

enum Readstatus{
    MN,
    A(i32), 
    B(i32),
    C(i32),
    END,
}


fn main(){
    env::set_var("RUST_BACKTRACE", "1");
    //--------read data--------
    println!("Reading data");

    let path = "./src/simplex_method_csvs/feasible0.csv";

    let csvdata : Csvdata = read_csv(path);
    let m: usize = csvdata.m as usize;
    let n: usize = csvdata.n as usize;


    let a : Array2df64 = Array::from_shape_vec((m, n) , csvdata.a).unwrap();
    let b : Array1df64 = Array::from(csvdata.b);
    let c : Array1df64 = Array::from(csvdata.c);

    println!("Complete reading data");
    let now = time::Instant::now();

    //println!("{:?}",ncm_enum((0..28).collect(),18).len());
    let answer = search_all_basis(a, b, c, m, n);

    //output elapsed time
    println!("time {:?}", now.elapsed());

    println!("x : {:?}, cTx :{}",answer.0,answer.1)
}

fn search_all_basis(a : Array2df64, b :Array1df64, c :Array1df64, m : usize, n :usize) -> (Array1df64,f64){

    let order_list: Vec<Vec<i8>> = ncm_enum((0..(n as i8)).collect(),m as u8);

    let order_len = order_list.len();
    println!("create nCm");

    let mut min_ctx = 1000000000.;
    let mut min_vector : Array1df64 = Array::zeros(m);


    let mut counter = 0;
    for order in order_list{
        //let ab = create_matrix_from_order(a, &order);
        let ab = create_matrix_from_order(&a, &order);

        match ab.factorize(){
            Ok(factrized) => {
                let f_ab = factrized;
                let x = f_ab.solve(&b).unwrap();

                let num_of_non_0 : i32 = x.to_vec()
                    .into_iter()
                    .map(|x| if x < EPS{1}else{0})
                    .sum();

                if num_of_non_0 == 0{
                    let restore_x = restore_x(x, &order, n);
                    let ctx = c.t().dot(&restore_x);

                    let parcent = (counter as f64) / (order_len as f64) * 100.;
                    println!("% : {}, order : {:?}, cTx : {}",parcent,order,ctx);
                    if ctx < min_ctx{
                        min_ctx = ctx;
                        min_vector = restore_x;
                    }
                }
            }
            _ => {}
        };

        counter += 1;
    }
    //
    (min_vector,min_ctx)
}


fn create_matrix_from_order(matrix :&Array2df64, order : &Vec<i8>) -> Array2df64{
    let m = matrix.nrows();
    let mut new_matrix : Array2df64 = Array::zeros((m,0));

    for i in order{
        let index = *i as usize;

        let index_row =  Array::from_shape_vec((m, 1) , matrix.index_axis(Axis(1), index).to_vec()).unwrap();

        new_matrix = concatenate![Axis(1), new_matrix, index_row];
    }

    new_matrix
}

fn restore_x(x : Array1df64, order : &Vec<i8>, n : usize) -> Array1df64{
    let mut new_x : Array1df64 = Array::zeros(n);

    let mut counter = 0;
    for i in order{
        let index = *i as usize;

        let x_counter = x[counter];
        new_x.slice_mut(s![index]).fill(x_counter);
        counter += 1;
    }

    new_x
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

        //println!("{:?}",divided);
        match status{
            Readstatus::MN => {
                m = divided[0].parse().expect("not a number");
                n = divided[1].parse().expect("not a number");

                status = Readstatus::A(-1);

                //println!("m : {},n : {}",m,n);
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
                    //println!("complete reading A");
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
                //println!("complete reading B");
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
                //println!("complete reading C");
            },
            Readstatus::END => {}
            
        }
    }

    Csvdata{
        m,n,a,b,c
    }
}