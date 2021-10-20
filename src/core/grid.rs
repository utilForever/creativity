/*!
# Grid
A contiguous growable two-Dimensional data structure.
Grid stores data in one-dimensional vector and behaves like two-dimensional vector.

# Examples

```
use creativity::*;

let grid = grid![[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]];
```
*/

use rand::prelude::{thread_rng, SliceRandom};

use std::cmp::{Eq, PartialEq};
use std::fmt::{Debug, Formatter, Result};
use std::iter::StepBy;
use std::mem::swap;
use std::ops::{Index, IndexMut};
use std::slice::Iter;

/// Macro for initializing a grid with values.
///
/// Similar to declaring a two-dimensional array in C/C++.
/// All rows must be the same length.
/// If all rows are not of the same length, panic occurs.
///
/// Example:
/// Example code that execute normally.
/// ```
/// use creativity::*;
///
/// let grid_u32: core::grid::Grid<u32> = grid![];
/// let grid_char = grid![['a', 'b']];
/// let grid_f32 = grid![[1.0, 2.0], [3.0, 4.0]];
/// ```
///
/// Example code that panic occurs.
/// ```ignore
/// use creativity::*;
///
/// let grid_error = grid![[1, 2, 3],
///                        [4, 5], // panic!
///                        [7, 8, 9]];
/// ```

#[macro_export]
macro_rules! grid {
    () => {
        $crate::core::grid::Grid::from_vec(0, 0, vec![]);
    };

    ( [$( $x:expr ),* ] ) => {
        {
            let vec = vec![$($x),*];
            let col = vec.len();

            $crate::core::grid::Grid::from_vec(1, col, vec)
        }
    };

    ( [$( $x0:expr ),*] $( $(,)+ [$( $x1:expr ),*] )* ) => {
        {
            let mut vec = Vec::new();

            $( vec.push($x0); )*

            let mut length: usize = vec.len();
            let cols: usize = length;

            $(
                $( vec.push($x1); )*
                length = vec.len();

                if length % cols != 0 {
                    panic!("All row size must be the same");
                }
            )*

            let rows: usize = length / cols;

            $crate::core::grid::Grid::from_vec(rows, cols, vec)
        }
    };
}

/// Elements of grid structure.
///
/// rows: The row size of the grid.
/// cols: The column size of the grid.
/// data: Store grid structure data in one-dimensional vector.
///
/// Store grid structure data using one-dimensional vector.
pub struct Grid<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> Grid<T> {
    /// Initialize grid with default value.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = core::grid::Grid::new(2, 2);
    /// assert_eq!(grid, grid![[0, 0], [0, 0]]);
    /// ```
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

    /// Initialize grid with parameter value.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = core::grid::Grid::init(2, 2, 1);
    ///
    /// assert_eq!(grid, grid![[1, 1], [1, 1]]);
    /// ```
    pub fn init(rows: usize, cols: usize, data: T) -> Grid<T> {
        Grid {
            rows,
            cols,
            data: vec![data; rows * cols],
        }
    }

    /// Initialize grid with parameter vector.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = core::grid::Grid::from_vec(2, 2, vec![1, 2, 3, 4]);
    ///
    /// assert_eq!(grid, grid![[1, 2], [3, 4]]);
    /// ```
    pub fn from_vec(rows: usize, cols: usize, mut vec: Vec<T>) -> Grid<T>
    where
        T: Default,
    {
        // if vector size is greater than grid size, panic occurs.
        // vector size is not greater than grid size.
        if rows * cols < vec.len() {
            panic!("Vector length must be not greater than rows * cols");
        }

        // if vector size is less than grid size, fill in the diff with default value.
        if rows * cols > vec.len() {
            let diff = rows * cols - vec.len();
            vec.append(&mut vec![T::default(); diff]);
        }

        // return Grid
        Grid {
            rows,
            cols,
            data: vec,
        }
    }

    /// Insert new values in a specific row.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2]];
    ///
    /// grid.insert_row(0, vec![3, 4]);
    ///
    /// assert_eq!(grid, grid![[3, 4], [1, 2]]);
    /// ```
    pub fn insert_row(&mut self, index: usize, insert_vec: Vec<T>) {
        // if vector size is not equal grid column size, panic occurs.
        // to insert, vector size and grid column size must be same.
        if self.cols != insert_vec.len() {
            panic!(
                "Not equal length - vector length({}) and grid column size({}) must be same.",
                insert_vec.len(),
                self.cols
            );
        }

        // if index is greater than grid row size, panic occurs.
        // to insert, index is not greater than grid row size.
        if index > self.rows {
            panic!("Out Of Bound - row size: {}.", index);
        }

        // calculate starting index to insert values.
        let start_idx = index * self.cols;

        // insert.
        for i in 0..self.cols {
            self.data.insert(start_idx + i, insert_vec[i].clone());
        }

        // resize.
        self.rows += 1;
    }

    /// Insert new values in a specific column.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1], [2]];
    ///
    /// grid.insert_col(1, vec![3, 4]);
    ///
    /// assert_eq!(grid, grid![[1, 3], [2, 4]]);
    /// ```
    pub fn insert_col(&mut self, index: usize, insert_vec: Vec<T>) {
        // if vector size is not equal grid row size, panic occurs.
        // to insert, vector size and grid row size must be same.
        if self.rows != insert_vec.len() {
            panic!(
                "Not equal length - vector size({}) and grid row size({}) must be same.",
                insert_vec.len(),
                self.rows
            );
        }

        // if index is greater than grid column size, panic occurs.
        // to insert, index is not greater than grid column size.
        if index > self.cols {
            panic!("Out Of Index - column index: {}", index);
        }

        for i in 0..self.rows {
            // calculate real index to insert.
            let data_idx = i * self.cols + index + i;
            self.data.insert(data_idx, insert_vec[i].clone());
        }

        // resize.
        self.cols += 1;
    }

    /// Remove values in specific row.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4], [5, 6]];
    ///
    /// grid.remove_row(1);
    ///
    /// assert_eq!(grid, grid![[1, 2], [5, 6]]);
    /// ```
    pub fn remove_row(&mut self, index: usize) {
        // if grid data is empty, panic occurs.
        // to remove, grid data is not empty.
        if self.data.is_empty() {
            panic!("Grid data is empty.");
        }

        // if index is greater than grid row size, panic ocuurs.
        // to remove, index is not greater than grid row size.
        if index >= self.rows {
            panic!("Out Of Index - row index: {}", index);
        }

        // calculate starting index to remove value.
        let remove_idx = index * self.cols;

        for _ in 0..self.cols {
            self.data.remove(remove_idx);
        }

        // resize.
        self.rows -= 1;
    }

    /// Remove values in specific column.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2, 3], [4, 5, 6]];
    ///
    /// grid.remove_col(1);
    ///
    /// assert_eq!(grid, grid![[1, 3], [4, 6]]);
    /// ```
    pub fn remove_col(&mut self, index: usize) {
        // if grid data is empty, panic occurs.
        // to remove, grid data is not empty.
        if self.data.is_empty() {
            panic!("Grid data is empty.");
        }

        // if index is greater than grid column size, panic ocuurs.
        // to remove, index is not greater than grid column size.
        if index >= self.cols {
            panic!("Out of Index - column index: {}", index);
        }

        for i in 0..self.rows {
            // calculate real index to remove value.
            let data_idx = i * self.cols + index - i;
            self.data.remove(data_idx);
        }

        // resize.
        self.cols -= 1;
    }

    // Functions: push_row, push_col functions.
    // Improvement:
    //  We can use extend_from_slice function in std::vec.
    //  User can use the push function with slice.

    /// Insert new values in the last row.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2]];
    ///
    /// grid.push_row(vec![3, 4]);
    ///
    /// assert_eq!(grid, grid![[1, 2], [3, 4]]);
    /// ```
    pub fn push_row(&mut self, push_vec: Vec<T>) {
        // if vector size is not equal grid column size, panic occurs.
        // to push, vector size and grid column size must be same.
        if self.cols != push_vec.len() {
            panic!(
                "Not equal length - vector length({}) and grid column size({}) must be same.",
                push_vec.len(),
                self.cols
            );
        }

        // push function insert value in the last row.
        // therefore, just need to push new values in grid data.
        for i in 0..self.cols {
            self.data.push(push_vec[i].clone());
        }

        // resize.
        self.rows += 1;
    }

    /// Insert new values in the last column.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1], [2]];
    ///
    /// grid.push_col(vec![3, 4]);
    ///
    /// assert_eq!(grid, grid![[1, 3], [2, 4]]);
    /// ```
    pub fn push_col(&mut self, push_vec: Vec<T>) {
        // if vector size is not equal grid row size, panic occurs.
        // to push, vector size and grid row size must be same.
        if self.rows != push_vec.len() {
            panic!(
                "Not equal length - vector length({}) and grid row size({}) must be same.",
                push_vec.len(),
                self.rows
            );
        }

        // the behavior of the function is similar to insert_col function.
        let index = self.cols;

        for i in 0..self.rows {
            let data_idx = i * self.cols + index + i;
            self.data.insert(data_idx, push_vec[i].clone());
        }

        // resize.
        self.cols += 1;
    }

    /// Remove values in the last row.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    ///
    /// grid.pop_row();
    ///
    /// assert_eq!(grid, grid![[1, 2]]);
    /// ```
    pub fn pop_row(&mut self) {
        // if grid data is empty, panic occurs.
        // to pop, grid data is not empty.
        if self.data.is_empty() {
            panic!("Grid data is empty.");
        }

        // pop function remove values in the last row.
        // so, just need to pop.
        for _ in 0..self.cols {
            self.data.pop();
        }

        // resize.
        self.rows -= 1;
    }

    /// Remove values in the last column.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2, 3], [4, 5, 6]];
    ///
    /// grid.pop_col();
    ///
    /// assert_eq!(grid, grid![[1, 2], [4, 5]]);
    /// ```
    pub fn pop_col(&mut self) {
        // if grid data is empty, panic occurs.
        // to pop, grid data is not empty.
        if self.data.is_empty() {
            panic!("Grid data is empty.");
        }

        // The behavior of the function is similar to the remove_col function.
        let index = self.cols - 1;

        for i in 0..self.rows {
            let data_idx = i * self.cols + index - i;
            self.data.remove(data_idx);
        }

        // resize.
        self.cols -= 1;
    }
}
