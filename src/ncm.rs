pub mod allncm{
    #[derive(Debug)]
    #[derive(Clone)]
    struct Cell{
        standing : Vec<i8>,
        num_vacant_seats : u8,
        sitting : Vec<i8>,
    }

    type Animal = Vec<Option<Cell>>;
    
    //use i8 (becase it is light)
    pub fn ncm_enum(vec : Vec<i8>,m : u8) -> Vec<Vec<i8>>{
        let single_cell = create_cell_from_vec(vec, m);
        let mut animal : Animal = vec![Some(single_cell)];

        let mut no_cell_has_vacant_seats = false;

        while no_cell_has_vacant_seats == false {

            no_cell_has_vacant_seats = true;

            for index in 0..animal.len(){

                let option_cell = animal[index].clone();

                match option_cell{
                    Some(cell)=>{
                        match divide_cell(cell){
                            None => {}
                            Some(two_cells) => {
        
                                if live_or_die(&two_cells.0) && live_or_die(&two_cells.1){
                                    animal[index] = Some(two_cells.0);
                                    animal.push(Some(two_cells.1));
                                }
                                else if live_or_die(&two_cells.0){
                                    animal[index] = Some(two_cells.0);
                                }
                                else if live_or_die(&two_cells.1){
                                    animal[index] = Some(two_cells.1);
                                }
                                else{
                                    animal[index] = None;
                                };
        
                                no_cell_has_vacant_seats = false;
                            }
                        }
                    }
                    None => {}
                }
                
            }
        }

        let mut good_vec = Vec::new();
        for option_cell in animal.into_iter().rev(){
            match option_cell{
                Some(cell) => {
                    if cell.num_vacant_seats == 0{
                        good_vec.push(cell.sitting);
                    }
                }
                None => {}
            }
            
        }

        good_vec
    }

    fn divide_cell(cell : Cell) -> Option<(Cell,Cell)>{
        if cell.num_vacant_seats > 0 && cell.standing.len() > 0{
            let mut standing = cell.standing;
            let mut sitting = cell.sitting;

            let tail = standing.pop().unwrap();
            
            let cell_0 = Cell{
                standing : standing.clone(),
                num_vacant_seats : cell.num_vacant_seats,
                sitting : sitting.clone(),
            };
            sitting.push(tail);
            let cell_1 = Cell{
                standing : standing,
                num_vacant_seats : cell.num_vacant_seats - 1,
                sitting : sitting,
            };
            Some((cell_0,cell_1))
        }
        else{
            None
        }
    }

    fn create_cell_from_vec(vec : Vec<i8>, m :u8)-> Cell{
        Cell { standing: vec, num_vacant_seats: m, sitting: vec![] }
    }

    fn live_or_die(cell: &Cell) -> bool{
        if cell.standing.len() < cell.num_vacant_seats as usize{
            false
        }
        else{
            true
        }
    }
}