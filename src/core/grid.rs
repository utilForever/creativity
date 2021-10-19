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
                    panic!("All rows must be the same length");
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
        // check grid size and vector size.
        // if vector size is greater than grid size, panic occurs.
        if rows * cols < vec.len() {
            panic!("Vector length is longer than rows * cols");
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
        // check agument vector size and column size.
        // if column size is not equal argument vector size, panic occurs.
        // to insert, column size and vector size must be same.
        if self.cols != insert_vec.len() {
            panic!("cols != insert vector length");
        }

        // check the row to insert(index) and grid row size.
        // if row size is less than index, panic occurs.
        // to insert, index is not greater than grid row size.
        if index > self.rows {
            panic!("out of bound");
        }

        // calculate real start index to insert.
        let start_idx = index * self.cols;

        // insert.
        for i in 0..self.cols {
            self.data.insert(start_idx + i, insert_vec[i].clone());
        }

        // increasing grid row size.
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
        // check agument vector size and row size.
        // if row size is not equal argument vector size, panic occurs.
        // to insert, row size and vector size must be same.
        if self.rows != insert_vec.len() {
            panic!("rows != insert vector length");
        }

        // check the column to insert(index) and grid column size.
        // if column size is less than index, panic occurs.
        // to insert, index is not greater than grid column size.
        if index > self.cols {
            panic!("out of bound");
        }

        for i in 0..self.rows {
            // calculate real index in insert.
            let data_idx = i * self.cols + index + i;
            self.data.insert(data_idx, insert_vec[i].clone());
        }

        // increasing grid column size
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
        if self.data.is_empty() {
            panic!("data is empty");
        }

        if index > self.rows {
            panic!("out of bound");
        }

        let remove_idx = index * self.cols;

        for _ in 0..self.cols {
            self.data.remove(remove_idx);
        }

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
        if self.data.is_empty() {
            panic!("data is empty");
        }

        if index > self.cols {
            panic!("out of bound");
        }

        for i in 0..self.rows {
            let data_idx = i * self.cols + index - i;
            self.data.remove(data_idx);
        }

        self.cols -= 1;
    }

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
        if self.cols != push_vec.len() {
            panic!("cols != push vector length");
        }

        for i in 0..self.cols {
            self.data.push(push_vec[i].clone());
        }

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
        if self.data.is_empty() {
            panic!("data is empty");
        }

        for _ in 0..self.cols {
            self.data.pop();
        }

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

    /// Return iterator over the grid.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    /// let mut iter = grid.iter();
    ///
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    /// Return iterator over the grid data in specific row.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    /// let mut iter = grid.iter_row(1);
    ///
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_row(&self, row: usize) -> Iter<T> {
        if row > self.rows {
            panic!("out of bound");
        }

        let start = row * self.cols;
        self.data[start..(start + self.cols)].iter()
    }

    /// Return iterator over the grid data in specific column.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    /// let mut iter = grid.iter_col(1);
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_col(&self, col: usize) -> StepBy<Iter<T>> {
        if col > self.cols {
            panic!("out of bound");
        }

        self.data[col..].iter().step_by(self.cols)
    }

    /// Reverse the grid data.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    ///
    /// grid.reverse();
    ///
    /// assert_eq!(grid, grid![[4, 3], [2, 1]]);
    /// ```
    pub fn reverse(&mut self) {
        self.data.reverse();
    }

    pub fn reverse_row(&mut self, row: usize) {
        if row > self.rows {
            panic!("out of bound");
        }

        let start_idx = row * self.cols;
        let mut reverse_row: Vec<T> = self.data[start_idx..start_idx + self.cols].to_vec();

        reverse_row.reverse();

        self.remove_row(row);
        self.insert_row(row, reverse_row);
    }

    pub fn reverse_col(&mut self, col: usize)
    where
        T: Default,
    {
        if col > self.cols {
            panic!("out of bound");
        }

        let mut reverse_col: Vec<T> = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            reverse_col.push(self.data[i * self.cols + col].clone());
        }

        reverse_col.reverse();

        self.remove_col(col);
        self.insert_col(col, reverse_col);
    }

    /// Replace a specific value in the grid with a new value.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2], [1, 2]];
    ///
    /// grid.replace(1, 3);
    ///
    /// assert_eq!(grid, grid![[3, 2], [1, 2]]);
    /// ```
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

    /// Replace all specific values in the grid with a new value.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 1], [1, 1]];
    ///
    /// grid.replace_all(1, 2);
    ///
    /// assert_eq!(grid, grid![[2, 2], [2, 2]]);
    /// ```
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

    /// Randomly shuffle the grid data.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    ///
    /// grid.shuffle();
    ///
    /// // not equal !
    /// assert_ne!(grid, grid![[1, 2], [3, 4]]);
    /// ```
    pub fn shuffle(&mut self) {
        self.data.shuffle(&mut thread_rng());
    }

    /// Transpose the grid.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    ///
    /// grid.transpose();
    ///
    /// assert_eq!(grid, grid![[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
    /// ```
    pub fn transpose(&mut self) {
        let mut vec = Vec::with_capacity(self.data.len());

        for i in 0..self.cols {
            for j in 0..self.rows {
                vec.push(self.data[j * self.cols + i].clone());
            }
        }

        self.data = vec.clone();
        swap(&mut self.rows, &mut self.cols);
    }

    /// Return the values that match the condition as a vector.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = grid![[1, 2, 3], [4, 5, 6]];
    /// let odd = grid.filter(|x| x % 2 == 1);
    ///
    /// assert_eq!(odd, vec![1, 3, 5]);
    /// ```
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

    /// Return the grid row size.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = grid![[1, 2, 3], [4, 5, 6]];
    /// let rows = grid.get_rows();
    ///
    /// assert_eq!(rows, 2);
    /// ```
    pub fn get_rows(&self) -> usize {
        self.rows
    }

    /// Return the grid column size.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = grid![[1, 2, 3], [4, 5, 6]];
    /// let cols = grid.get_cols();
    ///
    /// assert_eq!(cols, 3);
    /// ```
    pub fn get_cols(&self) -> usize {
        self.cols
    }

    /// Return the grid total size as a tuple.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid: core::grid::Grid<u32> = grid![[1, 2, 3], [4, 5, 6]];
    /// let size = grid.get_size();
    ///
    /// assert_eq!(size.0, 2);  // rows
    /// assert_eq!(size.1, 3);  // cols
    /// ```
    pub fn get_size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Check the grid is empty.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let grid1: core::grid::Grid<u32> = grid![];
    /// let grid2: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    ///
    /// assert_eq!(grid1.is_empty(), true);
    /// assert_eq!(grid2.is_empty(), false);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.cols == 0 && self.rows == 0 && self.data.is_empty()
    }

    /// Clear the grid.
    ///
    /// Example:
    /// ```
    /// use creativity::*;
    ///
    /// let mut grid: core::grid::Grid<u32> = grid![[1, 2], [3, 4]];
    ///
    /// grid.clear();
    ///
    /// assert_eq!(grid.is_empty(), true);
    /// ```
    pub fn clear(&mut self) {
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

impl<T: Eq> Eq for Grid<T> {}

impl<T: Debug> Debug for Grid<T> {
    #[allow(unused_must_use)]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "[");

        if self.cols > 0 && self.rows > 0 {
            for i in 0..self.rows {
                let start_idx = i * self.cols;

                write!(f, "{:?}", &self.data[start_idx..start_idx + self.cols]);

                if self.rows - i > 1 {
                    write!(f, ", ");
                }
            }
        }
        write!(f, "]")
    }
}

impl<T: Clone> Index<usize> for Grid<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        if idx < self.rows {
            let start_idx = idx * self.cols;
            &self.data[start_idx..start_idx + self.cols]
        } else {
            panic!("Out of bound");
        }
    }
}

impl<T: Clone> IndexMut<usize> for Grid<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        let start_idx = idx * self.cols;
        &mut self.data[start_idx..]
    }
}

mod test {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn new() {
        let grid: Grid<u32> = Grid::new(2, 3);

        assert_eq!(grid[0], [0, 0, 0]);
        assert_eq!(grid[1], [0, 0, 0]);
    }

    #[test]
    fn init() {
        let grid: Grid<u32> = Grid::init(2, 3, 1);

        assert_eq!(grid[0], [1, 1, 1]);
        assert_eq!(grid[1], [1, 1, 1]);
    }

    #[test]
    fn from_vec() {
        let grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);

        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!(grid[1], [4, 5, 6]);
    }

    #[test]
    #[should_panic]
    fn from_vec_less_than_vector_length() {
        Grid::from_vec(2, 2, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn from_vec_greater_than_vector_length() {
        let grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6]);

        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!(grid[1], [4, 5, 6]);
        assert_eq!(grid[2], [0, 0, 0]);
    }

    #[test]
    fn from_vec_zero() {
        let grid: Grid<u32> = Grid::from_vec(0, 0, vec![]);

        assert_eq!(grid.is_empty(), true);
    }

    #[test]
    fn insert_row() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.insert_row(1, vec![7, 8, 9]);

        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!(grid[1], [7, 8, 9]);
        assert_eq!(grid[2], [4, 5, 6]);
    }

    #[test]
    #[should_panic]
    fn insert_row_out_of_bound() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.insert_row(3, vec![7, 8, 9]);
    }

    #[test]
    #[should_panic]
    fn insert_row_not_eq_length() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.insert_row(1, vec![7, 8, 9, 10]);
    }

    #[test]
    fn insert_col() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        grid.insert_col(1, vec![7, 8, 9]);

        assert_eq!(grid[0], [1, 7, 2]);
        assert_eq!(grid[1], [3, 8, 4]);
        assert_eq!(grid[2], [5, 9, 6]);
    }

    #[test]
    #[should_panic]
    fn insert_col_out_of_bound() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        grid.insert_col(3, vec![7, 8, 9]);
    }

    #[test]
    #[should_panic]
    fn insert_col_not_eq_length() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        grid.insert_col(1, vec![7, 8, 9, 10]);
    }

    #[test]
    fn remove_row() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.remove_row(1);

        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!(grid[1], [7, 8, 9]);
    }

    #[test]
    #[should_panic]
    fn remove_row_out_of_bound() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.remove_row(3);
    }

    #[test]
    #[should_panic]
    fn remove_row_check_empty() {
        let mut grid: Grid<u32> = Grid::new(0, 0);
        grid.remove_row(1);
    }

    #[test]
    fn remove_col() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.remove_col(1);

        assert_eq!(grid[0], [1, 3]);
        assert_eq!(grid[1], [4, 6]);
        assert_eq!(grid[2], [7, 9]);
    }

    #[test]
    #[should_panic]
    fn remove_col_out_of_bound() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.remove_col(3);
    }

    #[test]
    #[should_panic]
    fn remove_col_check_empty() {
        let mut grid: Grid<u32> = Grid::new(0, 0);
        grid.remove_col(1);
    }

    #[test]
    fn push_row() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.push_row(vec![7, 8, 9]);

        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!(grid[1], [4, 5, 6]);
        assert_eq!(grid[2], [7, 8, 9]);
    }

    #[test]
    #[should_panic]
    fn push_row_not_eq_length() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.push_row(vec![7, 8, 9, 10]);
    }

    #[test]
    fn push_col() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        grid.push_col(vec![7, 8, 9]);

        assert_eq!(grid[0], [1, 2, 7]);
        assert_eq!(grid[1], [3, 4, 8]);
        assert_eq!(grid[2], [5, 6, 9]);
    }

    #[test]
    #[should_panic]
    fn push_col_not_eq_length() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        grid.push_col(vec![7, 8, 9, 10]);
    }

    #[test]
    fn pop_row() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.pop_row();

        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!(grid[1], [4, 5, 6]);
    }

    #[test]
    #[should_panic]
    fn pop_row_check_empty() {
        let mut grid: Grid<u32> = Grid::new(0, 0);
        grid.pop_row();
    }

    #[test]
    fn pop_col() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.pop_col();

        assert_eq!(grid[0], [1, 2]);
        assert_eq!(grid[1], [4, 5]);
        assert_eq!(grid[2], [7, 8]);
    }

    #[test]
    #[should_panic]
    fn pop_col_check_empty() {
        let mut grid: Grid<u32> = Grid::new(0, 0);
        grid.pop_col();
    }

    #[test]
    fn iter() {
        let grid: Grid<u32> = Grid::from_vec(2, 2, vec![1, 2, 3, 4]);
        let mut iter = grid.iter();

        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_row() {
        let grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut iter = grid.iter_row(1);

        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), Some(&5));
        assert_eq!(iter.next(), Some(&6));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_col() {
        let grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut iter = grid.iter_col(1);

        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&5));
        assert_eq!(iter.next(), Some(&8));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn reverse() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.reverse();

        assert_eq!(grid[0], [9, 8, 7]);
        assert_eq!(grid[1], [6, 5, 4]);
        assert_eq!(grid[2], [3, 2, 1]);
    }

    #[test]
    fn reverse_row() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        grid.reverse_row(1);

        assert_eq!(grid[0], [1, 2]);
        assert_eq!(grid[1], [4, 3]);
        assert_eq!(grid[2], [5, 6]);
    }

    #[test]
    #[should_panic]
    fn reverse_row_out_of_bound() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        grid.reverse_row(3);
    }

    #[test]
    fn reverse_col() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.reverse_col(1);

        assert_eq!(grid[0], [1, 5, 3]);
        assert_eq!(grid[1], [4, 2, 6]);
    }

    #[test]
    #[should_panic]
    fn reverse_col_out_of_bound() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.reverse_col(3);
    }

    #[test]
    fn replace() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 1, 1, 1, 1, 1, 1, 1, 1]);
        grid.replace(1, 2);

        assert_eq!(grid[0], [2, 1, 1]);
        assert_eq!(grid[1], [1, 1, 1]);
        assert_eq!(grid[2], [1, 1, 1]);
    }

    #[test]
    fn replace_not_change() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 1, 1, 1, 1, 1, 1, 1, 1]);
        grid.replace(2, 3);

        assert_eq!(grid[0], [1, 1, 1]);
        assert_eq!(grid[1], [1, 1, 1]);
        assert_eq!(grid[2], [1, 1, 1]);
    }

    #[test]
    fn replace_all() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 1, 1, 1, 1, 1, 1, 1, 1]);
        grid.replace_all(1, 2);

        assert_eq!(grid[0], [2, 2, 2]);
        assert_eq!(grid[1], [2, 2, 2]);
        assert_eq!(grid[2], [2, 2, 2]);
    }

    #[test]
    fn replace_all_not_change() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 1, 1, 1, 1, 1, 1, 1, 1]);
        grid.replace_all(2, 3);

        assert_eq!(grid[0], [1, 1, 1]);
        assert_eq!(grid[1], [1, 1, 1]);
        assert_eq!(grid[2], [1, 1, 1]);
    }

    #[test]
    fn shuffle() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.shuffle();

        assert_ne!(grid[0], [1, 2, 3]);
        assert_ne!(grid[1], [4, 5, 6]);
        assert_ne!(grid[2], [7, 8, 9]);
    }

    #[test]
    fn transpose_1() {
        let mut grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        grid.transpose();

        assert_eq!(grid[0], [1, 4, 7]);
        assert_eq!(grid[1], [2, 5, 8]);
        assert_eq!(grid[2], [3, 6, 9]);
    }

    #[test]
    fn transpose_2() {
        let mut grid: Grid<u32> = Grid::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        grid.transpose();

        assert_eq!(grid[0], [1, 4]);
        assert_eq!(grid[1], [2, 5]);
        assert_eq!(grid[2], [3, 6]);
    }

    #[test]
    fn filter() {
        let grid: Grid<u32> = Grid::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let even: Vec<u32> = grid.filter(|x| x % 2 == 0);
        let odd: Vec<u32> = grid.filter(|x| x % 2 == 1);

        assert_eq!(even, vec![2, 4, 6, 8]);
        assert_eq!(odd, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn get_rows() {
        let grid: Grid<u32> = Grid::new(2, 3);
        let row = grid.get_rows();

        assert_eq!(row, 2);
    }

    #[test]
    fn get_cols() {
        let grid: Grid<u32> = Grid::new(2, 3);
        let col = grid.get_cols();

        assert_eq!(col, 3);
    }

    #[test]
    fn get_size() {
        let grid: Grid<u32> = Grid::new(2, 3);
        let size = grid.get_size();

        assert_eq!(size.0, 2);
        assert_eq!(size.1, 3);
    }

    #[test]
    fn is_empty_true() {
        let grid: Grid<u32> = Grid::new(0, 0);
        let empty = grid.is_empty();

        assert_eq!(empty, true);
    }

    #[test]
    fn is_empty_false() {
        let grid: Grid<u32> = Grid::new(1, 1);
        let empty = grid.is_empty();

        assert_eq!(empty, false);
    }

    #[test]
    fn clear() {
        let mut grid: Grid<u32> = Grid::init(3, 3, 1);
        grid.clear();

        let empty = grid.is_empty();

        assert_eq!(empty, true);
    }

    #[test]
    fn eq() {
        let grid1: Grid<u32> = Grid::from_vec(2, 2, vec![1, 2, 3, 4]);
        let grid2: Grid<u32> = Grid::from_vec(2, 2, vec![1, 2, 3, 4]);

        assert_eq!(grid1 == grid2, true);
    }

    #[test]
    fn eq_empty() {
        let grid1: Grid<u32> = Grid::new(0, 0);
        let grid2: Grid<u32> = Grid::new(0, 0);

        assert_eq!(grid1 == grid2, true);
    }

    #[test]
    fn ne() {
        let grid1: Grid<u32> = Grid::from_vec(2, 2, vec![1, 2, 3, 4]);
        let grid2: Grid<u32> = Grid::from_vec(2, 2, vec![1, 2, 3, 5]);

        assert_ne!(grid1, grid2);
    }

    #[test]
    fn ne_diff_row_and_col() {
        let grid1: Grid<u32> = Grid::from_vec(2, 2, vec![1, 2, 3, 4]);
        let grid2: Grid<u32> = Grid::from_vec(1, 4, vec![1, 2, 3, 4]);

        assert_ne!(grid1, grid2);
    }

    #[test]
    fn ne_full_and_empty() {
        let grid1: Grid<u32> = Grid::from_vec(2, 2, vec![1, 2, 3, 4]);
        let grid2: Grid<u32> = Grid::new(0, 0);

        assert_ne!(grid1, grid2);
    }

    #[test]
    fn index() {
        let grid: Grid<u32> = Grid::init(3, 3, 1);

        assert_eq!(grid[0][0], 1);
    }

    #[test]
    #[should_panic]
    fn index_out_of_bound() {
        let grid: Grid<u32> = Grid::init(3, 3, 1);
        grid[20][0];
    }

    #[test]
    fn index_set() {
        let mut grid: Grid<u32> = Grid::init(3, 3, 1);
        grid[0][0] = 2;

        assert_eq!(grid[0][0], 2);
    }

    #[test]
    fn macro_1() {
        let grid: Grid<u32> = grid![];

        assert_eq!(grid.is_empty(), true);
    }

    #[test]
    fn macro_2() {
        let grid: Grid<u32> = grid![[1, 2, 3, 4]];
        let rows = grid.get_rows();
        let cols = grid.get_cols();

        assert_eq!(rows, 1);
        assert_eq!(cols, 4);
        assert_eq!(grid[0], [1, 2, 3, 4]);
    }

    #[test]
    fn macro_3() {
        let grid: Grid<u32> = grid![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let rows = grid.get_rows();
        let cols = grid.get_cols();

        assert_eq!(rows, 3);
        assert_eq!(cols, 3);
        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!(grid[1], [4, 5, 6]);
        assert_eq!(grid[2], [7, 8, 9]);
    }

    #[test]
    #[should_panic]
    fn macro_diff_col_size() {
        grid![[1, 2, 3], [4, 5], [7, 8, 9]];
    }
}
