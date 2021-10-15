use rand::prelude::{thread_rng, SliceRandom};

use std::cmp::{Eq, PartialEq};

use std::mem::swap;

use std::iter::StepBy;
use std::slice::Iter;

pub struct Grid<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> Grid<T> {
    pub fn new(rows: usize, cols: usize) -> Grid<T>
    where
        T: Default,
    {
        Grid {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    pub fn init(rows: usize, cols: usize, data: T) -> Grid<T> {
        Grid {
            rows,
            cols,
            data: vec![data; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, mut vec: Vec<T>) -> Grid<T>
    where
        T: Default,
    {
        if rows * cols > vec.len() {
            let diff = rows * cols - vec.len();
            vec.append(&mut vec![T::default(); diff]);
        }

        Grid {
            rows,
            cols,
            data: vec,
        }
    }

    pub fn insert_col(&mut self, index: usize, insert_vec: Vec<T>) {
        if self.rows != insert_vec.len() {
            panic!("rows != insert vector length");
        }

        if index > self.cols {
            panic!("out of bound");
        }

        for i in 0..self.rows {
            let data_idx = i * self.cols + index + i;
            self.data.insert(data_idx, insert_vec[i].clone());
        }

        self.cols += 1;
    }

    pub fn insert_row(&mut self, index: usize, insert_vec: Vec<T>) {
        if self.cols != insert_vec.len() {
            panic!("cols != insert vector length");
        }

        if index > self.rows {
            panic!("out of bound");
        }

        let start_idx = index * self.cols;

        for i in 0..self.cols {
            self.data.insert(start_idx + i, insert_vec[i].clone());
        }

        self.rows += 1;
    }

    pub fn remove_col(&mut self, index: usize) {
        if self.data.is_empty() {
            panic!("data is empty");
        }

        for i in 0..self.rows {
            let data_idx = i * self.cols + index - i;
            self.data.remove(data_idx);
        }

        self.cols -= 1;
    }

    pub fn remove_row(&mut self, index: usize) {
        if self.data.is_empty() {
            panic!("data is empty");
        }

        let remove_idx = index * self.cols;

        for _ in 0..self.cols {
            self.data.remove(remove_idx);
        }

        self.rows -= 1;
    }

    pub fn push_col(&mut self, push_vec: Vec<T>) {
        if self.rows != push_vec.len() {
            panic!("rows != push vector length");
        }

        let index = self.cols;

        for i in 0..self.rows {
            let data_idx = i * self.cols + index + i;
            self.data.insert(data_idx, push_vec[i].clone());
        }

        self.cols += 1;
    }

    pub fn push_row(&mut self, push_vec: Vec<T>) {
        if self.cols != push_vec.len() {
            panic!("cols != push vector length");
        }

        for i in 0..self.cols {
            self.data.push(push_vec[i].clone());
        }

        self.rows += 1;
    }

    pub fn pop_col(&mut self) {
        if self.data.is_empty() {
            panic!("data is empty");
        }

        let index = self.cols - 1;

        for i in 0..self.rows {
            let data_idx = i * self.cols + index - i;
            self.data.remove(data_idx);
        }

        self.cols -= 1;
    }

    pub fn pop_row(&mut self) {
        if self.data.is_empty() {
            panic!("data is empty");
        }

        for _ in 0..self.cols {
            self.data.pop();
        }

        self.rows -= 1;
    }

    pub fn iter_col(&self, col: usize) -> StepBy<Iter<T>> {
        self.data[col..].iter().step_by(self.cols)
    }

    pub fn iter_row(&self, row: usize) -> Iter<T> {
        let start = row * self.cols;
        self.data[start..(start + self.cols)].iter()
    }

    pub fn reverse(&mut self) {
        self.data.reverse();
    }

    pub fn replace(&mut self, old_data: T, new_data: T)
    where
        T: PartialEq,
    {
        let index = self.rows * self.cols;

        for i in 0..index {
            if self.data[i] == old_data {
                self.data[i] = new_data;
                break;
            }
        }
    }

    pub fn replace_all(&mut self, old_data: T, new_data: T)
    where
        T: PartialEq,
    {
        let index = self.rows * self.cols;

        for i in 0..index {
            if self.data[i] == old_data {
                self.data[i] = new_data.clone();
            }
        }
    }

    pub fn shuffle(&mut self) {
        self.data.shuffle(&mut thread_rng());
    }

    pub fn transpose(&mut self) {
        let mut vec = Vec::with_capacity(self.data.len());

        for i in 0..self.cols {
            for j in 0..self.rows {
                vec.push(self.data[j * self.rows + i].clone());
            }
        }

        self.data = vec.clone();
        swap(&mut self.rows, &mut self.cols);
    }

    pub fn filter<F>(&self, condition: F) -> Vec<T>
    where
        F: FnMut(&T) -> bool,
    {
        self.data
            .clone()
            .into_iter()
            .filter(condition)
            .collect::<Vec<_>>()
    }

    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_cols(&self) -> usize {
        self.cols
    }

    pub fn get_size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn is_empty(&self) -> bool {
        self.cols == 0 && self.rows == 0 && self.data.is_empty()
    }

    pub fn clear(&mut self) {
        /*
         * grid.clear()
         *
         * Clear grid
         */

        self.rows = 0;
        self.cols = 0;
        self.data.clear();
    }
}

impl<T: Clone> Clone for Grid<T> {
    fn clone(&self) -> Self {
        Grid {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<T: Eq> PartialEq for Grid<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}
