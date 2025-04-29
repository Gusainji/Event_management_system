-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Feb 19, 2025 at 05:02 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `event_management`
--

-- --------------------------------------------------------

--
-- Table structure for table `categories`
--

CREATE TABLE `categories` (
  `id` int(11) NOT NULL,
  `name` varchar(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `categories`
--

INSERT INTO `categories` (`id`, `name`) VALUES
(1, 'NextJS'),
(2, 'AI'),
(3, 'Angular'),
(4, 'Tech'),
(5, 'UI/UX'),
(6, 'NodeJS');

-- --------------------------------------------------------

--
-- Table structure for table `events`
--

CREATE TABLE `events` (
  `id` int(11) NOT NULL,
  `name` varchar(64) NOT NULL,
  `description` text NOT NULL,
  `auther` varchar(64) NOT NULL,
  `start_date` bigint(20) NOT NULL,
  `end_date` bigint(20) NOT NULL,
  `duration` bigint(20) NOT NULL,
  `category_id` int(11) NOT NULL,
  `is_online` tinyint(1) NOT NULL,
  `location` varchar(64) DEFAULT NULL,
  `is_free` tinyint(1) NOT NULL,
  `price` float DEFAULT NULL,
  `image` varchar(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `events`
--

INSERT INTO `events` (`id`, `name`, `description`, `auther`, `start_date`, `end_date`, `duration`, `category_id`, `is_online`, `location`, `is_free`, `price`, `image`) VALUES
(2, 'Tech Innovators Summit', 'A conference bringing together tech leaders to discuss emerging innovations.', 'vipulk', 1740637800000, 1741392000000, 120, 3, 0, 'New York Convention Center', 0, 2000, '1739025632374-737692654.png'),
(3, 'Digital Marketing Masterclass', 'A workshop on the latest trends and strategies in digital marketing.', 'vipulk', 1740033000000, 1743033600000, 120, 5, 1, '', 0, 200, '1739026719016-573715836.png'),
(4, '3D Printing for Engineers', 'Learn the latest advancements in industrial 3D printing technology.', 'vipulk', 1740033000000, 1743033600000, 120, 6, 1, '', 1, 0, '1739026820877-991577962.png'),
(5, 'IoT & Smart Cities Expo', 'Discover how IoT is shaping the future of smart cities.', 'sachin', 1739370600000, 1740096000000, 60, 4, 0, 'Dubai Smart Tech Center', 0, 10000, '1739099949340-328926551.png'),
(7, 'Machine Learning Bootcamp', 'Hands-on training in machine learning algorithms and applications.', 'sachin', 1740799800000, 1738972800000, 60, 2, 1, '', 1, 0, '1739598199336-690982729.png');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `username` varchar(64) NOT NULL,
  `name` varchar(64) NOT NULL,
  `password` varchar(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`username`, `name`, `password`) VALUES
('sachin', 'Sachin Gusain', '$2b$10$tB82tRWaXiKdH27gNKzUe.LOZ20hMgwT/C3JPamY6sk7FtQf3udCu'),
('vipulk', 'Vipul Kumar', '$2b$10$/3sUMKmjDVpPfnGtroi9HeOaOMggtrMvgZIYyX2YBNWlhbw9Vt6Xm');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `events`
--
ALTER TABLE `events`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`username`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `categories`
--
ALTER TABLE `categories`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- AUTO_INCREMENT for table `events`
--
ALTER TABLE `events`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
